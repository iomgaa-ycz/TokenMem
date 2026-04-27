"""OOD 知识生成的格式转换测试。"""

from tools.build_ood_knowledge import convert_medqa, convert_arc, convert_mmlu


class TestConvertMedqa:
    def test_basic(self):
        row = {
            "id": "test-00000",
            "sent1": "A patient presents with...",
            "sent2": "",
            "ending0": "Option A text",
            "ending1": "Option B text",
            "ending2": "Option C text",
            "ending3": "Option D text",
            "label": 1,
        }
        result = convert_medqa(row, idx=0)
        assert result["id"] == "medqa_00000"
        assert result["dataset"] == "medqa"
        assert result["question"] == "A patient presents with..."
        assert result["options"] == {
            "A": "Option A text",
            "B": "Option B text",
            "C": "Option C text",
            "D": "Option D text",
        }
        assert result["correct_letter"] == "B"

    def test_label_0_maps_to_A(self):
        row = {
            "id": "test-00001",
            "sent1": "Q",
            "sent2": "",
            "ending0": "a",
            "ending1": "b",
            "ending2": "c",
            "ending3": "d",
            "label": 0,
        }
        assert convert_medqa(row, idx=1)["correct_letter"] == "A"


class TestConvertArc:
    def test_basic(self):
        row = {
            "id": "Mercury_7175875",
            "question": "An astronomer observes...",
            "choices": {
                "text": [
                    "Density decreases",
                    "Years longer",
                    "Days shorter",
                    "Gravity stronger",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        }
        result = convert_arc(row, idx=0)
        assert result["id"] == "arc_00000"
        assert result["dataset"] == "arc"
        assert result["correct_letter"] == "C"
        assert result["options"]["C"] == "Days shorter"


class TestConvertMmlu:
    def test_basic(self):
        row = {
            "question": "Find the degree...",
            "subject": "abstract_algebra",
            "choices": ["0", "4", "2", "6"],
            "answer": 1,
        }
        result = convert_mmlu(row, idx=42)
        assert result["id"] == "mmlu_00042"
        assert result["dataset"] == "mmlu"
        assert result["correct_letter"] == "B"
        assert result["options"]["B"] == "4"
