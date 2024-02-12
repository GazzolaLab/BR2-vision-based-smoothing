import numpy as np
import h5py


# string_names = ['Paul', 'John', 'Anna']
# float_heights = [5.9, 5.7,  6.1]
# int_ages = [27, 31, 33]
# numpy_data = [ np.array([5.4, 6.7, 8.8]), 
#                np.array([3.1, 58.4, 66.4]),
#                np.array([4.7, 5.1, 4.2])  ] 
# 
# # Create empty record array with 3 rows
# ds_dtype = [('name','S50'), ('height',float), ('ages',int), ('numpy_data', float, (3,) ) ]
# ds_arr = np.recarray((3,),dtype=ds_dtype)
# # load list data to record array by field name
# ds_arr['name'] = np.asarray(string_names)
# ds_arr['height'] = np.asarray(float_heights)
# ds_arr['ages'] = np.asarray(int_ages)
# ds_arr['numpy_data'] = np.asarray(numpy_data)
# 
# with h5py.File('SO_59483094.h5', 'w') as h5f:
# # load data to dataset my_ds1 using recarray
#     dset = h5f.create_dataset('my_ds1', data=ds_arr, maxshape=(None) )
# # load data to dataset my_ds2 by lists/field names
#     dset = h5f.create_dataset('my_ds2', dtype=ds_dtype, shape=(100,), maxshape=(None) )
#     dset['name',0:3] = np.asarray(string_names)
#     dset['height',0:3] = np.asarray(float_heights)
#     dset['ages',0:3] = np.asarray(int_ages)
#     dset['numpy_data',0:3] = np.asarray(numpy_data)

# Tracking data:
