def zero_intervals(n_cols, nnz_sorted):
    intervals = []
    prev = -1
    for c in nnz_sorted:
        if c > prev + 1:
            intervals.append((prev + 1, c - 1))
        prev = c
    if prev < n_cols - 1:
        intervals.append((prev + 1, n_cols - 1))
    return intervals
