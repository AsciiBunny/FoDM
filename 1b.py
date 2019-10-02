from typing import List, Set, FrozenSet

data = {
    'Rubber Soul':          [1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    'Revolver':             [1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    'Sgt. Pepper\'s LHCB':  [0, 0, 0, 0, 1, 0, 1, 1, 0, 0], #
    'Magical Mystery Tour': [1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    'The Beatles':          [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    'Yellow Submarine':     [0, 1, 0, 0, 0, 0, 1, 0, 1, 0], #
    'Abbey Road':           [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], #
    'Let It Be':            [1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
}


def get_transactions(data):
    transactions = []
    for i in range(10):
        transaction = set()
        for key in data:
            if data[key][i] == 1:
                transaction.add(key)
        transactions.append(transaction)
    return transactions


def get_frequency(transactions: List[Set], itemset):
    containing = 0
    for transaction in transactions:
        if transaction.issuperset(itemset):
            containing += 1
    return containing / len(transactions)


def prune(transactions: List[Set], itemsets: Set[FrozenSet], r: float):
    filtered_itemsets = set()
    for itemset in itemsets:
        if get_frequency(transactions, itemset) >= r:
            filtered_itemsets.add(itemset)
    return filtered_itemsets


def generate_candidates(previous_itemsets: Set[FrozenSet]):
    new_itemsets = set()
    for a in previous_itemsets:
        for b in previous_itemsets:
            if len(a.difference(b)) == 1 and len(b.difference(a)) == 1:
                new_itemsets.add(a.union(b))
    return new_itemsets


def find_frequent_itemsets(transactions: List[Set], r: float):
    c = list()
    c.append(set([frozenset([key]) for key in data.keys()]))
    l = list()
    l.append(prune(transactions, c[0], r))
    k = 0
    while len(l[k]) > 0:
        c.append(generate_candidates(l[k]))
        l.append(prune(transactions, c[k + 1], r))
        k += 1
    return [set(item) for sublist in l for item in sublist]


transactions = get_transactions(data)
frequent_itemsets = find_frequent_itemsets(transactions, 0.4)
print(frequent_itemsets)
print(max(frequent_itemsets, key=len))
