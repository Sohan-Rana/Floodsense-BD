from floodsense.cli import main


def test_train_and_predict_cli(tmp_path, capsys):
    model_dir = str(tmp_path / "m")
    assert main(["--model-dir", model_dir, "train", "--years", "2"]) == 0
    out = capsys.readouterr().out
    assert "roc_auc" in out

    code = main(
        [
            "--model-dir", model_dir, "predict", "--district", "sunamganj",
            "--rain-24h", "70", "--rain-72h", "180", "--rain-7d", "320", "--forecast-48h", "120",
            "--river-ratio", "1.1",
        ]
    )
    assert code == 0
    assert "Flood probability" in capsys.readouterr().out


def test_districts_command(capsys):
    assert main(["districts"]) == 0
    assert "sunamganj" in capsys.readouterr().out
