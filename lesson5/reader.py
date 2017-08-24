# -*-coding:utf-8-*-#
import collections

def get_poetrys(poetry_file="./poetry.txt"):
    poetrys = []
    with open(poetry_file, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            try:
                title, content = line.strip().split(":")
                content = content.replace(" ", "")
                if '_' in content or '(' in content or \
                '（' in content or '《' in content or \
                '[' in content :
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                poetrys.append(content)
            except Exception as e:
                pass
        return poetrys

def build_dataset():
    poetrys = get_poetrys()
    poetrys = sorted(poetrys, key=lambda line: len(line))
    print("唐诗总数:", len(poetrys))
    words = []
    for poetry in poetrys:
        words += [word for word in poetry]
    counter = collections.Counter(words)
    # 从大到小排序
    counter_pairs = sorted(counter.items(), key=lambda x: -x[1])
    # 从counter中解压，并获取当中的词(不重复)
    words, _ = zip(*counter_pairs)
    words = words[:len(words)] + (" ", )
    # word -> id
    dictionary = dict(zip(words, range(len(words))))
    # id -> word
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    poetry_vectors = [[dictionary[word] for word in poetry] for poetry in poetrys]
    return dictionary, poetry_vectors, reversed_dictionary
