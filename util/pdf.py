# tgb - 10/15/2019 - Utilities that can come in handy when calculating PDFs of various quantities

# From middle of bins to edges
def bin_mid_to_edge(binm):
    bine = 0.5*(binm[:-1]+binm[1:]) # bin_edges[1:-1]
    return np.concatenate(([bine[0]-(bine[2]-bine[1])],bine,[bine[-1]+(bine[2]-bine[1])]))

# From index to date in a xarray 
def date_index(time_array,time_index):
    return time_array[time_index].values

# From edges to middle of bins
def edgTObin(edges):
    return 0.5*(edges[1:]+edges[:-1])

# From date to index in a xarray 
def index_date(time_array,date_string):
    return [i for i, x in enumerate(time_array.sel({'time':date_string})==time_array) if x]

