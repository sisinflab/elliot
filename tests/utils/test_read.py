import pytest

from elliot.utils.folder import file_full_name
from elliot.utils.read import Reader

from tests.params import reader_folder, reader_path


reader = Reader()

read_tabular = reader.read_tabular
read_sequence_tabular = reader.read_sequence_tabular
read_folder = reader.read_folder
read_model = reader.read_model
read_json = reader.read_json


class TestReadTabular:

    def test_no_header(self):
        path = reader_path("data_no_header")

        df = read_tabular(path, header=False)

        assert list(df.columns) == [0, 1, 2]
        assert df.shape == (2, 3)

    def test_with_header(self):
        path = reader_path("data")

        df = read_tabular(path, header=True)

        assert list(df.columns) == ["userId", "itemId", "rating"]
        assert df.shape[0] == 1

    def test_columns_positional_selection_and_reorder(self):
        path = reader_path("data")

        df = read_tabular(path, header=True, columns=[2, 0])

        assert list(df.columns) == ["rating", "userId"]
        assert df.iloc[0].tolist() == [5, 1]

    def test_columns_semantic_selection(self):
        path = reader_path("data")

        df = read_tabular(path, header=True, columns=["itemId", "rating"])

        assert list(df.columns) == ["itemId", "rating"]

    def test_no_columns_matched_returns_empty(self):
        path = reader_path("data")

        df = read_tabular(path, header=True, columns=["nonexistent"])

        assert df.empty

    def test_datatypes_applied(self):
        path = reader_path("data")

        df = read_tabular(
            path, header=True, columns=["userId", "rating"], datatypes={"rating": "float"}
        )

        assert df["rating"].dtype == float

    def test_empty_file_returns_empty_dataframe(self):
        path = reader_path("empty")

        df = read_tabular(path, header=False, columns=["userId", "itemId"])

        assert df.empty
        assert list(df.columns) == ["userId", "itemId"]

    def test_callback_fn_applied(self):
        path = reader_path("data")

        df = read_tabular(
            path, header=True,
            callback_fn=lambda d, add: d.assign(rating=d["userId"] + add), add=10
        )

        assert list(df["rating"]) == [11]


class TestReadSequenceTabular:

    def test_wide_no_header(self):
        path = reader_path("wide_no_header")

        df = read_sequence_tabular(path, format="wide", header=False, columns=["userId"])

        assert list(df.columns) == ["userId", "itemId", "_sourceRow"]
        assert df.shape[0] == 5
        assert list(df[df["userId"] == "1"]["itemId"]) == ["1", "2", "3"]
        assert list(df[df["userId"] == "2"]["itemId"]) == ["4", "5"]
        assert list(df["_sourceRow"]) == [0, 0, 0, 1, 1]

    def test_wide_with_header(self):
        path = reader_path("wide")

        df = read_sequence_tabular(path, format="wide", header=True, columns=["userId"])

        assert df.shape[0] == 5
        assert list(df[df["userId"] == "1"]["itemId"]) == ["1", "2", "3"]

    def test_wide_track_source_rows_disabled(self):
        path = reader_path("wide")

        df = read_sequence_tabular(path, format="wide", header=True, columns=["userId"], track_source_rows=False)

        assert df.shape[0] == 5
        assert "_sourceRow" not in df.columns

    def test_inline_with_timestamp(self):
        path = reader_path("inline")

        df = read_sequence_tabular(
            path, format="inline", header=True, columns=["userId", "sequence", "timestamp"]
        )

        assert list(df.columns) == ["userId", "timestamp", "_sourceRow", "itemId"]
        assert df.shape[0] == 5
        u1 = df[df["userId"] == 1]
        assert list(u1["itemId"]) == ["1", "2", "3"]
        assert (u1["timestamp"] == 10).all()
        assert list(df["_sourceRow"]) == [0, 0, 0, 1, 1]

    def test_inline_without_timestamp(self):
        path = reader_path("inline")

        df = read_sequence_tabular(path, format="inline", header=True, columns=["userId", "sequence"])

        assert list(df.columns) == ["userId", "_sourceRow", "itemId"]
        assert df.shape[0] == 5

    def test_inline_custom_sequence_sep(self):
        path = reader_path("inline_custom_seq_sep")

        df = read_sequence_tabular(
            path, format="inline", header=True, columns=["userId", "sequence"], sequence_sep=","
        )

        assert list(df["itemId"]) == ["1", "2", "3"]

    def test_inline_missing_required_column(self):
        path = reader_path("inline_missing_required_column")

        df = read_sequence_tabular(path, format="inline", header=True, columns=["userId", "sequence"])

        assert df.empty

    @pytest.mark.parametrize("format", ["wide", "inline"])
    def test_empty_file(self, format):
        path = reader_path("empty")

        df = read_sequence_tabular(path, format=format, header=False, columns=["userId"])

        assert df.empty
        assert list(df.columns) == ["userId", "itemId"]

    def test_invalid_format(self):
        path = reader_path("wide_no_header")

        with pytest.raises(ValueError):
            read_sequence_tabular(path, format="invalid_format")


class TestReadFolder:

    def test_lists_only_files(self):
        path = reader_path(folder="folder")

        files = read_folder(path)

        assert {file_full_name(f) for f in files} == {"a.tsv", "b.csv"}

    def test_filter_by_pattern(self):
        path = reader_path(folder="folder")

        files = read_folder(path, patterns="*a.tsv")

        assert [file_full_name(f) for f in files] == ["a.tsv"]

    def test_filter_by_extension(self):
        path = reader_path(folder="folder")

        files = read_folder(path, ext=[".tsv"])

        assert [file_full_name(f) for f in files] == ["a.tsv"]


# class TestReadMapping:
#
#     def test_basic_mapping(self, tmp_path):
#         path = tmp_path / "map.tsv"
#         path.write_text("1\t10\t20\n2\t30\n")
#
#         mapping = Reader().read_mapping(str(path), dtype="int")
#
#         assert set(mapping[1]) == {10, 20}
#         assert mapping[2] == [30]
#
#     def test_bracketed_identifier(self, tmp_path):
#         path = tmp_path / "map.tsv"
#         path.write_text("[42]\t100\t200\n")
#
#         mapping = Reader().read_mapping(str(path), dtype="int")
#
#         assert set(mapping[42]) == {100, 200}
#
#     def test_remove_duplicates_false_keeps_order_and_repeats(self, tmp_path):
#         path = tmp_path / "map.tsv"
#         path.write_text("1\t10\t10\t20\n")
#
#         mapping = Reader().read_mapping(str(path), dtype="int", remove_duplicates=False)
#
#         assert mapping[1] == [10, 10, 20]
#
#     def test_callback_fn_applied(self, tmp_path):
#         path = tmp_path / "map.tsv"
#         path.write_text("1\t10\n")
#
#         mapping = Reader().read_mapping(
#             str(path), dtype="int", callback_fn=lambda m, extra: {**m, "extra": extra}, extra=True
#         )
#
#         assert mapping["extra"] is True
#         assert mapping[1] == [10]


def test_read_model():
    model_obj = {"weights": [1, 2, 3]}

    loaded = read_model(reader_folder, model_name="mymodel")

    assert loaded == model_obj


def test_read_json():
    path = reader_path("data", ext="json")

    data = read_json(path)

    assert data == {"a": 1, "b": [1, 2, 3]}


class TestDiscoverNpyIds:

    def test_lists_ids_and_sniffs_shape(self, tmp_path):
        import numpy as np

        np.save(tmp_path / "1.npy", np.array([1.0, 2.0]))
        np.save(tmp_path / "2.npy", np.array([3.0, 4.0]))

        ids, id_map, shape = reader.discover_npy_ids(str(tmp_path))

        assert ids == {1, 2}
        assert set(id_map.keys()) == {1, 2}
        assert shape == (2,)

    def test_falsy_folder_returns_empty(self):
        assert reader.discover_npy_ids(None) == (set(), {}, None)


class TestReadNpy:

    def test_loads_full_array(self, tmp_path):
        import numpy as np

        path = tmp_path / "1.npy"
        np.save(path, np.array([1.0, 2.0, 3.0]))

        loaded = reader.read_npy(str(path))

        assert list(loaded) == [1.0, 2.0, 3.0]
        assert not isinstance(loaded, np.memmap)

    def test_mmap_mode_returns_memmap_without_full_copy(self, tmp_path):
        import numpy as np

        path = tmp_path / "1.npy"
        np.save(path, np.array([1.0, 2.0, 3.0]))

        loaded = reader.read_npy(str(path), mmap_mode="r")

        assert isinstance(loaded, np.memmap)
        assert list(loaded) == [1.0, 2.0, 3.0]


class TestReadTriplesAsTuples:

    def test_tsv_splits_on_tab(self, tmp_path):
        path = tmp_path / "triples.tsv"
        path.write_text("e1\trel1\te2\ne2\trel2\te3\n")

        triples = reader.read_triples_as_tuples(str(path))

        assert triples == [("e1", "rel1", "e2"), ("e2", "rel2", "e3")]

    def test_non_tsv_splits_on_whitespace(self, tmp_path):
        path = tmp_path / "triples.txt"
        path.write_text("e1 rel1 e2\ne2 rel2 e3\n")

        triples = reader.read_triples_as_tuples(str(path))

        assert triples == [("e1", "rel1", "e2"), ("e2", "rel2", "e3")]


if __name__ == '__main__':
    pytest.main()
