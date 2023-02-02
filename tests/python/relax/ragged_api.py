b = tir.Var("b", "int64")
r_b = relax.Var("r_b", relax.RaggedDimType(False, b, None, None))


parent_dim = n
ind_ptr = ind_ptr
r_l = relax.Var("r_l", None , relax.RaggedDimType(True, None, parent_dim, ind_ptr)) # ragged dim

dims = [b, r_l]
grouping = [[0, 1]]
l0 = relax.Var("l0", [dims, grouping], relax.RaggedLayout(2)) # layout


data_src = 
v0 = relax.Var("v0",  [data_src, layout], relax.RaggedDynTensorType(2, dtype)) # ragged tensor