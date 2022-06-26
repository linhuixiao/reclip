"""Use spatial relations extracted from the parses."""

from typing import Dict, Any, Callable, List, Tuple, NamedTuple
from numbers import Number
from collections import defaultdict
from overrides import overrides
import numpy as np
import spacy
from spacy.tokens.token import Token
from spacy.tokens.span import Span
from argparse import Namespace

from .ref_method import RefMethod
from lattice import Product as L
from heuristics import Heuristics
from entity_extraction import Entity, expand_chunks


def get_conjunct(ent, chunks, heuristics: Heuristics) -> Entity:
    """If an entity represents a conjunction of two entities, pull them apart."""
    head = ent.head.root  # Not ...root.head. Confusing names here.
    if not any(child.text == "and" for child in head.children):
        return None
    for child in head.children:
        if child.i in chunks and head.i is not child.i:
            return Entity.extract(child, chunks, heuristics)
    return None


class Parse(RefMethod):
    """An REF method that extracts and composes predicates, relations, and superlatives from a dependency parse.
        从依赖项解析中提取并组合谓词、关系和最高级的REF方法。

    The process is as follows:
        1. Use spacy to parse the document.                               1. 使用 spacy 去解析文档/句子
        2. Extract a semantic entity tree from the parse.                 2. 从 parse 中 提取语义实体树
        3. Execute the entity tree to yield a distribution over boxes.    3. 执行实体树以生成与box相对应的分布。
    """

    # spacy 是工业级的NLP处理库，下述代码表示用 spacy 导入英文词汇解析库 en_core_web_sm 生成 NLP 工具对象 nlp
    nlp = spacy.load('en_core_web_sm')
    '''初始化'''
    def __init__(self, args: Namespace = None):
        self.args = args
        self.box_area_threshold = args.box_area_threshold
        self.baseline_threshold = args.baseline_threshold
        self.temperature = args.temperature
        self.superlative_head_only = args.superlative_head_only
        self.expand_chunks = args.expand_chunks
        self.branch = not args.parse_no_branch  # TODO 注意；此处已经加了一次 False
        self.possessive_expand = not args.possessive_no_expand  # 根据默认推算， self.possessive_expand == true

        # Lists of keyword heuristics to use.
        # 初始化启发式谓词关系解析器配置，并没有实际开始计算
        self.heuristics = Heuristics(args)

        # Metrics for debugging relation extraction behavor.
        self.counts = defaultdict(int)

    # 重载 execute
    @overrides
    def execute(self, caption: str, env: "Environment") -> Dict[str, Any]:
        """Construct an `Entity` tree from the parse and execute it to yield a distribution over boxes."""
        """从解析中构造一个“实体”树,并执行它以产生一个关于boxes的分布
            caption 是一个完整的句子 """
        # Start by using the full caption, as in Baseline.
        '''调用env，过滤小box, 增加其他box形式，执行CLIP！！'''
        # TODO: 输出一个关于 box 的概率分布，这个box 指的是提取的 box，还是 gt box？？
        probs = env.filter(caption, area_threshold=self.box_area_threshold, softmax=True)

        # Extend the baseline using parse stuff.
        doc = self.nlp(caption)
        head = self.get_head(doc)
        chunks = self.get_chunks(doc)

        # 默认 False
        if self.expand_chunks:
            chunks = expand_chunks(doc, chunks)

        # 提取实体
        entity = Entity.extract(head, chunks, self.heuristics)
        # print("Entity 1: ", entity)
        # eg. "the man in yellow coat"
        # Entity(head=the man, relations=[([in], Entity(head=yellow coat, relations=[], superlatives=[]))], superlatives=[])
        # eg. 2: a hot dog with chili on top
        # Entity(head=a hot dog, relations=[([with], Entity(head=chili, relations=[], superlatives=[[on, top]]))], superlatives=[])

        # If no head noun is found, take the first one.
        if entity is None and len(list(doc.noun_chunks)) > 0:
            head = list(doc.noun_chunks)[0]
            entity = Entity.extract(head.root.head, chunks, self.heuristics)
            self.counts["n_0th_noun"] += 1
            # print("\n Entity 2: ", entity)

        # If we have found some head noun, filter based on it.
        # python的与或非的逻辑还和C++不一样，C++ 是 and 是与，or 是或，and 需要同时为1 则为 1. entity 正常为true，关键看后半部分。 1 or 0
        # 如果中间 any() 部分为 true 则为 true，如果any() 部分为False，则需要看 not self.branch，而not self.branch 一直为 False， 所以只看any()部分即可
        # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。
        # TODO: 后面那句话怎么理解？？ 不仅要有 entity， 还要满足句子中的 token 信息 还要在启发式、比较级最高级的词库里。如果没有则不执行启发式
        # if entity is not None and (any(any(token.text in h.keywords for h in self.heuristics.relations+self.heuristics.superlatives) for token in doc) or not self.branch):
        if 0:
            # TODO: 最核心的计算，到底是怎么算的，数据的维度又是怎么样的，怎么能和 CLIP 直接点对点相乘？
            ent_probs, texts = self.execute_entity(entity, env, chunks)
            # TODO: L.meet 也是 2 组 数组相乘啊
            print("\n\n This is the parser result:")
            print("CLIP result: ", probs)
            print("parser result: ", ent_probs)
            probs = L.meet(probs, ent_probs)
            print("clip * parser result: ", probs)
        else:
            texts = [caption]
            self.counts["n_full_expr"] += 1

        self.counts["n_total"] += 1
        pred = np.argmax(probs)
        # print("CLIP result after softmax: ", probs)
        # print("predicts: ", pred)
        return {
            "probs": probs,
            "pred": pred,
            "box": env.boxes[pred],
            "texts": texts
        }

    # TODO: 没太懂这个函数，这才是整个代码最核心的部分
    # eg. 2: a hot dog with chili on top
    # Entity(head=a hot dog, relations=[([with], Entity(head=chili, relations=[], superlatives=[[on, top]]))], superlatives=[])
    def execute_entity(self,
                       ent: Entity,
                       env: "Environment",
                       chunks: Dict[int, Span],
                       root: bool = True,) -> np.ndarray:
        """Execute an `Entity` tree recursively, yielding a distribution over boxes."""
        """递归地执行“实体”树，在boxes上生成分布。"""
        self.counts["n_rec"] += 1
        probs = [1, 1]
        head_probs = probs

        # Only use relations if the head baseline isn't certain.
        # 只有在头部基线不确定的情况下才使用关系。如果 box 数量为 1 就别用了。
        if len(probs) == 1 or len(env.boxes) == 1:
            return probs, [ent.text]

        m1, m2 = probs[:2]  # probs[(-probs).argsort()[:2]]
        text = ent.text  # a hot dog
        rel_probs = []
        # 默认 true
        if self.baseline_threshold == float("inf") or m1 < self.baseline_threshold * m2:
            self.counts["n_rec_rel"] += 1
            for tokens, ent2 in ent.relations:
                print("\n\ntokens: ", tokens)
                print("ent2: ", ent2)
                # tokens:  [with]
                # ent2: Entity(head=chili, relations=[], superlatives=[[on, top]])
                self.counts["n_rel"] += 1
                rel = None
                # Heuristically decide which spatial relation is represented. 启发式地决定哪个空间关系被表示。
                """遍历普通的空间关系：left，west，bigger，larger，inside"""
                for heuristic in self.heuristics.relations:
                    if any(tok.text in heuristic.keywords for tok in tokens):
                        print("this is entered relation: ", heuristic.keywords)
                        # TODO: callback 是啥意思？？
                        rel = heuristic.callback(env)
                        self.counts[f"n_rel_{heuristic.keywords[0]}"] += 1
                        print("rel is：", rel)
                        break
                # Filter and normalize by the spatial relation. 根据空间关系进行过滤和归一化。
                if rel is not None:
                    # 如果找到了空间关系，则再迭代执行一遍
                    print("this is entered rel: ", rel)
                    probs2 = self.execute_entity(ent2, env, chunks, root=False)
                    # 得到的分布结果再累乘回来
                    events = L.meet(np.expand_dims(probs2, axis=0), rel)
                    new_probs = L.join_reduce(events)
                    rel_probs.append((ent2.text, new_probs, probs2))
                    continue

                # This case specifically handles "between", which takes two noun arguments. 这种情况专门处理“between”，它有两个名词参数
                """遍历二元关系"""
                rel = None
                for heuristic in self.heuristics.ternary_relations:
                    if any(tok.text in heuristic.keywords for tok in tokens):
                        print("this is entered ternary: ", heuristic.keywords)
                        rel = heuristic.callback(env)
                        self.counts[f"n_rel_{heuristic.keywords[0]}"] += 1
                        break
                if rel is not None:
                    ent3 = get_conjunct(ent2, chunks, self.heuristics)
                    if ent3 is not None:
                        probs2 = self.execute_entity(ent2, env, chunks, root=False)
                        probs2 = np.expand_dims(probs2, axis=[0, 2])
                        probs3 = self.execute_entity(ent3, env, chunks, root=False)
                        probs3 = np.expand_dims(probs3, axis=[0, 1])
                        events = L.meet(L.meet(probs2, probs3), rel)
                        new_probs = L.join_reduce(L.join_reduce(events))
                        probs = L.meet(probs, new_probs)
                    continue

                # Otherwise, treat the relation as a possessive relation. 否则，将这种关系视为占有关系。
                if not self.args.no_possessive:  # True
                    if self.possessive_expand:  # True, 默认进入此分支
                        print("进入1，ent2.head：", ent2.head)
                        # ent2.head: chili
                        # ent.expand() 的实现见 entity_extraction, 实现原理未搞清楚
                        text = ent.expand(ent2.head)  # 如此一来，头有2个
                    else:
                        print("进入2")
                        text += f' {" ".join(tok.text for tok in tokens)} {ent2.text}'
                    # TODO: 又执行了 CLIP ！！！
                    print("possessive_expand text:", text)
                    # possessive_expand text: a hot dog with chili on top，没啥变换
                    poss_probs = self._filter(text, env, root=root, expand=.3)
            # TODO: 又执行了 CLIP ！！！多次调用 CLIP, token 遍历完了再过一遍 CLIP
            print("ent.relations text: ", text)
            # ent.relations text: a hot dog with chili on top
            probs = self._filter(text, env, root=root)
            texts = [text]
            return_probs = [(probs.tolist(), probs.tolist())]
            for (ent2_text, new_probs, ent2_only_probs) in rel_probs:
                probs = L.meet(probs, new_probs)
                probs /= probs.sum()
                texts.append(ent2_text)
                return_probs.append((probs.tolist(), ent2_only_probs.tolist()))

        # Only use superlatives if thresholds work out. 只有在阈值成立的情况下才使用最高级。
        """最高级"""
        m1, m2 = probs[(-probs).argsort()[:2]]
        print("m1, m2, thresholds：", m1, m2, self.baseline_threshold)
        # m1, m2, thresholds： 0.9999999 1.44498e-07 inf
        if m1 < self.baseline_threshold * m2:  # 无穷大，默认是需要用的
            print("execute superlatives 1")
            self.counts["n_rec_sup"] += 1
            for tokens in ent.superlatives:
                print("execute superlatives 2")
                self.counts["n_sup"] += 1
                sup = None
                for heuristic_index, heuristic in enumerate(self.heuristics.superlatives):
                    print("execute superlatives 3")
                    if any(tok.text in heuristic.keywords for tok in tokens):
                        print("this is entered superlatives: ", heuristic.keywords)
                        texts.append('sup:'+' '.join([tok.text for tok in tokens if tok.text in heuristic.keywords]))
                        sup = heuristic.callback(env)
                        self.counts[f"n_sup_{heuristic.keywords[0]}"] += 1
                        break
                if sup is not None:
                    # Could use `probs` or `head_probs` here?
                    precond = head_probs if self.superlative_head_only else probs
                    probs = L.meet(np.expand_dims(precond, axis=1)*np.expand_dims(precond, axis=0), sup).sum(axis=1)
                    probs = probs / probs.sum()
                    return_probs.append((probs.tolist(), None))

        if root:
            assert len(texts) == len(return_probs)
            return probs, (texts, return_probs, tuple(str(chunk) for chunk in chunks.values()))
        return probs

    def get_head(self, doc) -> Token:
        """Return the token that is the head of the dependency parse."""
        for token in doc:
            if token.head.i == token.i:
                return token
        return None

    def get_chunks(self, doc) -> Dict[int, Any]:
        """Return a dictionary mapping sentence indices to their noun chunk."""
        chunks = {}
        for chunk in doc.noun_chunks:
            for idx in range(chunk.start, chunk.end):
                chunks[idx] = chunk
        return chunks

    @overrides
    def get_stats(self) -> Dict[str, Number]:
        """Summary statistics that have been tracked on this object."""
        stats = dict(self.counts)
        n_rel_caught = sum(v for k, v in stats.items() if k.startswith("n_rel_"))
        n_sup_caught = sum(v for k, v in stats.items() if k.startswith("n_sup_"))
        stats.update({
            "p_rel_caught": n_rel_caught / (self.counts["n_rel"] + 1e-9),
            "p_sup_caught": n_sup_caught / (self.counts["n_sup"] + 1e-9),
            "p_rec_rel": self.counts["n_rec_rel"] / (self.counts["n_rec"] + 1e-9),
            "p_rec_sup": self.counts["n_rec_sup"] / (self.counts["n_rec"] + 1e-9),
            "p_0th_noun": self.counts["n_0th_noun"] / (self.counts["n_total"] + 1e-9),
            "p_full_expr": self.counts["n_full_expr"] / (self.counts["n_total"] + 1e-9),
            "avg_rec": self.counts["n_rec"] / self.counts["n_total"],
        })
        return stats

    def _filter(self,
                caption: str,
                env: "Environment",
                root: bool = False,
                expand: float = None,
               ) -> np.ndarray:
        """Wrap a filter call in a consistent way for all recursions."""
        """对所有递归以一致的方式包装 filter 的调用，整个函数也就换了caption，其他都一样"""
        kwargs = {
            "softmax": not self.args.sigmoid,
            "temperature": self.args.temperature,
        }
        if root:
            return env.filter(caption, area_threshold=self.box_area_threshold, **kwargs)
        else:
            return env.filter(caption, **kwargs)

