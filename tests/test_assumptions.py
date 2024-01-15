import pytest
from dask import bag as db


def test_bag_sequence_order():
    '''
    Tests the order of the dask bag.
    See https://stackoverflow.com/questions/56177374/does-dask-bag-from-sequence-preserve-order
    '''
    bag = db.from_sequence([1,2,3,4,5])
    square = lambda i: i**2
    assert bag.map(square).compute() == [i**2 for i in [1,2,3,4,5]]
