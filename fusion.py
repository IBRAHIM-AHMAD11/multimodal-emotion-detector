def fuse_emotions(results):
    votes = {}
    for val in results.values():
        votes[val] = votes.get(val, 0) + 1
    return max(votes, key=votes.get)