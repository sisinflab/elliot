import pytest
import pandas as pd
import torch
from pathlib import Path

from elliot.utils.write import Writer
from elliot.utils.folder import path_joiner, check_path, check_dir

from tests.utils.test_read import read_tabular, read_sequence_tabular, read_model, read_json
from tests.params import writer_folder, writer_path

check_dir(writer_folder, replace=True)

writer = Writer()

write_tabular = writer.write_tabular
write_sequence_tabular = writer.write_sequence_tabular
write_model = writer.write_model
write_recommendations = writer.write_recommendations
write_dict_as_table = writer.write_dict_as_table
write_json = writer.write_json


class TestWriteTabular:

    def test_basic_write_no_header(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = writer_path("out")

        write_tabular(df, path)

        assert Path(path).read_text() == "1\tx\n2\ty\n"

    def test_with_header_bool(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        path = writer_path("out_header_bool")

        write_tabular(df, path, header=True)

        assert Path(path).read_text() == "a\tb\n1\tx\n"

    def test_with_header_alias_list(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        path = writer_path("out_header")

        write_tabular(df, path, header=["col1", "col2"])

        assert Path(path).read_text() == "col1\tcol2\n1\tx\n"

    def test_header_alias_mismatch_falls_back_to_no_header(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        path = writer_path("out_header_mismatch")

        write_tabular(df, path, header=["only_one"])

        assert Path(path).read_text() == "1\tx\n"

    def test_columns_positional_selection_and_reorder(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        path = writer_path("out_columns_positional")

        write_tabular(df, path, columns=[2, 0])

        assert Path(path).read_text() == "3\t1\n"

    def test_columns_semantic_selection(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        path = writer_path("out_columns_semantic")

        write_tabular(df, path, columns=["b"])

        assert Path(path).read_text() == "2\n"

    def test_no_columns_matched_writes_empty_file(self):
        df = pd.DataFrame({"a": [1]})
        path = writer_path("out_no_columns_matched")

        write_tabular(df, path, columns=["missing"])

        assert Path(path).read_text() == ""

    def test_custom_separator(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        path = writer_path("out_custom_separator")

        write_tabular(df, path, sep=",")

        assert Path(path).read_text() == "1,2\n"

    def test_callback_fn_applied(self):
        df = pd.DataFrame({"a": [1]})
        path = writer_path("out_callback")
        received = {}

        write_tabular(
            df, path, callback_fn=lambda d, flag: received.update(flag=flag, shape=d.shape), flag=True
        )

        assert received == {"flag": True, "shape": (1, 1)}


class TestWriteSequenceTabular:

    def test_wide_orders_items_by_timestamp(self):
        df = pd.DataFrame({
            "userId": ["1", "1", "1", "2", "2"],
            "itemId": ["1", "2", "3", "4", "5"],
            "timestamp": [1, 2, 0, 1, 0],
        })
        path = writer_path("wide")

        write_sequence_tabular(df, path, format="wide")

        assert Path(path).read_text() == "1\t3\t1\t2\n2\t5\t4\n"

    def test_wide_writes_one_line_per_session(self):
        df = pd.DataFrame({
            "userId": ["1", "1", "1", "1", "2"],
            "itemId": ["1", "2", "3", "4", "5"],
            "sessionId": [0, 0, 1, 1, 0],
            "timestamp": [0, 1, 2, 3, 0],
        })
        path = writer_path("wide_session_id")

        write_sequence_tabular(df, path, format="wide")

        assert Path(path).read_text() == "1\t1\t2\n1\t3\t4\n2\t5\n"

    def test_wide_without_timestamp_groups_by_user_only(self):
        df = pd.DataFrame({
            "userId": ["1", "1", "2"],
            "itemId": ["1", "2", "3"]
        })
        path = writer_path("wide_no_timestamp")

        write_sequence_tabular(df, path, format="wide")

        assert Path(path).read_text() == "1\t1\t2\n2\t3\n"

    def test_wide_header_alias_list(self):
        df = pd.DataFrame({"userId": ["1"], "itemId": ["1"]})
        path = writer_path("wide_header")

        write_sequence_tabular(df, path, format="wide", header=["user", "items"])

        assert Path(path).read_text() == "user\titems\n1\t1\n"

    def test_wide_header_alias_mismatch_falls_back_to_no_header(self):
        df = pd.DataFrame({"userId": ["1"], "itemId": ["1"]})
        path = writer_path("wide_header_mismatch")

        write_sequence_tabular(df, path, format="wide", header=["only_one"])

        assert Path(path).read_text() == "1\t1\n"

    def test_inline_without_timestamp_column(self):
        df = pd.DataFrame({
            "userId": ["1", "1", "2"],
            "itemId": ["1", "2", "3"]
        })
        path = writer_path("inline_no_timestamp")

        write_sequence_tabular(df, path, format="inline", header=True)

        assert Path(path).read_text() == "userId\tsequence\n1\t1 2\n2\t3\n"

    def test_inline_custom_sequence_sep(self):
        df = pd.DataFrame({
            "userId": ["1", "1"],
            "itemId": ["1", "2"],
            "timestamp": [0, 1]
        })
        path = writer_path("inline_custom_sequence_sep")

        write_sequence_tabular(df, path, format="inline", sequence_sep=",")

        assert "1,2" in Path(path).read_text()

    def test_inline_header_alias_list(self):
        df = pd.DataFrame({"userId": ["1"], "itemId": ["1"]})
        path = writer_path("inline_header")

        write_sequence_tabular(df, path, format="inline", header=["U", "S"])

        assert Path(path).read_text() == "U\tS\n1\t1\n"

    def test_inline_columns_excludes_timestamp_from_output(self):
        df = pd.DataFrame({"userId": ["1"], "itemId": ["1"], "timestamp": [5]})
        path = writer_path("inline_columns")

        write_sequence_tabular(df, path, format="inline", columns=["userId", "sequence"])

        assert Path(path).read_text() == "1\t1\n"

    def test_invalid_format(self):
        df = pd.DataFrame({"userId": ["1"], "itemId": ["1"]})
        path = writer_path("invalid_format")

        with pytest.raises(ValueError):
            write_sequence_tabular(df, path, format="invalid_format")


class TestWriteModel:

    def test_writes_model_file_with_expected_name(self):
        model_obj = {"weights": [1, 2, 3]}

        write_model(model_obj, writer_folder, model_name="mymodel")

        path = writer_path("best-weights-mymodel", ext="pth")
        assert check_path(path)
        assert torch.load(path) == model_obj

    def test_creates_missing_save_folder(self):
        save_folder = path_joiner(writer_folder, "nested", "models")

        write_model({"weights": [1]}, save_folder, model_name="mymodel")

        assert check_path(path_joiner(save_folder, "best-weights-mymodel.pth"))

    def test_custom_extension(self):
        write_model({"weights": [1]}, writer_folder, model_name="mymodel", ext=".pt")

        assert check_path(writer_path("best-weights-mymodel", ext="pt"))


class TestWriteRecommendations:

    def test_writes_top_k_recommendations(self):
        recs = {"1": [("1", 0.9), ("2", 0.5)], "2": [("3", 0.7)]}

        write_recommendations(recs, writer_folder, model_name="mymodel")

        path = writer_path("mymodel")
        assert check_path(path)
        with open(path) as f:
            assert f.read() == "1\t1\t0.9\n1\t2\t0.5\n2\t3\t0.7\n"

    def test_it_suffix_included_in_filename(self):
        recs = {"1": [("1", 0.9)]}

        write_recommendations(recs, writer_folder, model_name="mymodel", it=3)

        assert check_path(writer_path("mymodel_it=3"))

    def test_creates_missing_save_folder(self):
        save_folder = path_joiner(writer_folder, "nested", "recs")

        write_recommendations({"1": [("1", 0.5)]}, save_folder, model_name="mymodel")

        assert check_path(path_joiner(save_folder, "mymodel.tsv"))


# class TestWriteResults:
#
#     def test_writes_tabular_when_not_triplets(self, tmp_path):
#         results = {5: {"model_a": {"nDCG": 0.5, "Precision": 0.3}}}
#         save_folder = str(tmp_path)
#
#         Writer().write_results(results, save_folder)
#
#         files = list(Path(save_folder).glob("rec_cutoff_5*.tsv"))
#         assert len(files) == 1
#         assert files[0].read_text() == "model_a\t0.5\t0.3\n"
#
#     def test_writes_triplets_when_requested(self, tmp_path):
#         results = {5: {"model_a": {"nDCG": 0.5, "Precision": 0.3}}}
#         save_folder = str(tmp_path)
#
#         Writer().write_results(results, save_folder, triplets=True)
#
#         files = list(Path(save_folder).glob("triplets_rec_cutoff_5*.tsv"))
#         assert len(files) == 1
#         assert files[0].read_text() == "model_a\tnDCG\t0.5\nmodel_a\tPrecision\t0.3\n"
#
#     def test_skips_empty_cutoffs(self, tmp_path):
#         save_folder = str(tmp_path)
#
#         Writer().write_results({5: {}}, save_folder)
#
#         assert list(Path(save_folder).iterdir()) == []
#
#
# class TestWriteTimes:
#
#     def test_writes_timing_data(self, tmp_path):
#         save_folder = str(tmp_path)
#
#         Writer().write_times({"model_a": {"train_time": 1.23}}, save_folder)
#
#         files = list(Path(save_folder).glob("rec_training_time*.tsv"))
#         assert len(files) == 1
#         assert files[0].read_text() == "model_a\t1.23\n"
#
#
# class TestWriteTrials:
#
#     def test_writes_json_format(self, tmp_path):
#         save_folder = str(tmp_path)
#
#         Writer().write_trials({"model_a": [{"loss": 0.1}, {"loss": 0.2}]}, save_folder)
#
#         files = list(Path(save_folder).glob("trials_model_a*.json"))
#         assert len(files) == 1
#         assert json.loads(files[0].read_text()) == [{"loss": 0.1}, {"loss": 0.2}]
#
#     def test_writes_tabular_format(self, tmp_path):
#         save_folder = str(tmp_path)
#
#         Writer().write_trials({"model_a": [{"loss": 0.1}]}, save_folder, frmt="tabular")
#
#         files = list(Path(save_folder).glob("trials_model_a*.tsv"))
#         assert len(files) == 1
#         assert files[0].read_text() == "0.1\n"
#
#
# class TestWriteParams:
#
#     def test_writes_params_json(self, tmp_path):
#         save_folder = str(tmp_path)
#         params = [{"default_validation_cutoff": 10, "lr": 0.01}]
#
#         Writer().write_params(params, save_folder)
#
#         files = list(Path(save_folder).glob("bestmodelparams_cutoff_10*.json"))
#         assert len(files) == 1
#         assert json.loads(files[0].read_text()) == params
#
#
# class TestWriteStatisticalResults:
#
#     def test_writes_stat_results(self, tmp_path):
#         save_folder = str(tmp_path)
#
#         Writer().write_statistical_results({5: {"model_a": [0.1, 0.2]}}, save_folder, stat_test="wilcoxon")
#
#         files = list(Path(save_folder).glob("stat_wilcoxon_cutoff_5*.tsv"))
#         assert len(files) == 1
#         assert files[0].read_text() == "0.1\n0.2\n"
#
#     def test_skips_when_empty(self, tmp_path):
#         save_folder = str(tmp_path)
#
#         Writer().write_statistical_results({}, save_folder)
#
#         assert list(Path(save_folder).iterdir()) == []


class TestWriteDictAsTable:

    def test_writes_dict_as_table(self):
        data = {"model_a": {"nDCG": 0.5}, "model_b": {"nDCG": 0.6}}

        write_dict_as_table(data, writer_folder, file_name="summary")

        path = writer_path("summary")
        assert Path(path).read_text() == "model_a\t0.5\nmodel_b\t0.6\n"


class TestWriteJson:

    def test_writes_json_with_default_indent(self):
        path = writer_path("data", ext="json")

        write_json({"a": 1, "b": [1, 2]}, path)

        assert Path(path).read_text() == '{\n  "a": 1,\n  "b": [\n    1,\n    2\n  ]\n}'

    def test_custom_indent(self):
        path = writer_path("data", ext="json")

        write_json({"a": 1}, path, indent=4)

        assert Path(path).read_text() == '{\n    "a": 1\n}'


class TestReaderWriterInteroperability:
    """Round-trip checks exercising Writer's output through Reader, to ensure the two agree
    on layout for methods whose files are meant to be read back.
    """

    def test_tabular_roundtrip_with_header(self):
        df = pd.DataFrame({"userId": ["1", "2"], "rating": [5, 3]})
        path = writer_path("out")

        write_tabular(df, path, header=True)
        back = read_tabular(path, header=True)
    
        assert list(back.columns) == ["userId", "rating"]
        assert back["rating"].tolist() == [5, 3]

    def test_sequence_wide_header_is_skippable_on_read(self):
        df = pd.DataFrame({"userId": ["1", "1"], "itemId": ["1", "2"], "timestamp": [0, 1]})
        path = writer_path("wide")

        write_sequence_tabular(df, path, format="wide", header=True)
        back = read_sequence_tabular(path, format="wide", header=True, columns=["userId"])

        assert list(back["itemId"]) == ["1", "2"]

    def test_sequence_wide_roundtrip_without_timestamp(self):
        df = pd.DataFrame({"userId": ["1", "1", "2"], "itemId": ["1", "2", "3"]})
        path = writer_path("wide")

        write_sequence_tabular(df, path, format="wide")
        back = read_sequence_tabular(path, format="wide", header=False, columns=["userId"])

        assert back.shape[0] == 3

    def test_sequence_inline_orders_items_and_keeps_first_timestamp(self):
        df = pd.DataFrame({
            "userId": ["1", "1", "1", "2", "2"],
            "itemId": ["2", "3", "1", "5", "4"],
            "timestamp": [1, 2, 0, 1, 0],
        })
        path = writer_path("inline")

        write_sequence_tabular(df, path, format="inline", header=True)
        back = read_sequence_tabular(
            path, format="inline", header=True, columns=["userId", "sequence", "timestamp"]
        )

        u1 = back[back["userId"] == 1]
        u2 = back[back["userId"] == 2]
        assert list(u1["itemId"]) == ["1", "2", "3"]
        assert (u1["timestamp"] == 0).all()
        assert list(u2["itemId"]) == ["4", "5"]
        assert (u2["timestamp"] == 0).all()

    def test_sequence_inline_writes_one_row_per_session(self):
        df = pd.DataFrame({
            "userId": ["1", "1", "1", "1"],
            "itemId": ["1", "2", "3", "4"],
            "sessionId": [0, 0, 1, 1],
            "timestamp": [0, 1, 2, 3],
        })
        path = writer_path("inline")

        write_sequence_tabular(df, path, format="inline", header=True)
        back = read_sequence_tabular(
            path, format="inline", header=True, columns=["userId", "sequence", "timestamp"]
        )

        assert back["timestamp"].nunique() == 2
        sessions = back.groupby("timestamp")["itemId"].apply(list).sort_index()
        assert list(sessions) == [["1", "2"], ["3", "4"]]

    def test_model_roundtrip(self):
        model_obj = {"weights": [1, 2, 3]}

        write_model(model_obj, writer_folder, model_name="mymodel")
        loaded = read_model(writer_folder, model_name="mymodel")

        assert loaded == model_obj

    def test_json_roundtrip(self):
        path = writer_path("data", ext="json")

        write_json({"a": 1, "b": [1, 2, 3]}, path)
        loaded = read_json(path)

        assert loaded == {"a": 1, "b": [1, 2, 3]}


if __name__ == '__main__':
    pytest.main()
