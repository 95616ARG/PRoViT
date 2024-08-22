import os, sys; sys.path.append(os.getcwd())
import pathlib
working_dir = pathlib.Path(__file__).parent
sys.path.append(pathlib.Path(__file__).parents[2].as_posix())
import datasets

"""To run these tests, run: 

python PRoViT/provit/test_provit.py {path}

where {path} is the directory containing the ImageNet datasets.
"""

def main(arg):
    test_repair_sets(arg)
    test_gen_sets(arg)
    print("All tests passed!")
    return

def test_repair_sets(path):
    n = 5
    repair_sets = datasets.get_repair_sets(n, metric=3, path=path, seed=0)
    assert len(repair_sets) == 100

    set_of_labels = set()
    for d in repair_sets:
        assert len(d) == (n*5)
        for _, label in d:
            set_of_labels.add(label.item())
    assert len(set_of_labels) == n
    return

def test_gen_sets(path):
    n = 3
    gen_sets = datasets.get_gen_sets(n, metric=3, path=path, seed=0)
    assert len(gen_sets) == 100

    set_of_labels = set()
    for d in gen_sets:
        assert len(d) == (n*45)
        for _, label in d:
            set_of_labels.add(label.item())
    assert len(set_of_labels) == n
    return


if __name__ == "__main__":
    main(sys.argv[1])