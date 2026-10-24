# SmartVision — Machine Translation (TranslationModel) Module

The **TranslationModel** subproject translates medication packaging instructions, timing guidance, and audio responses between Telugu, Hindi, and English for the SmartVision smart-glasses system.

---

## 1. Safety-Centric Medical Translation

Translating medication instructions is high-risk. A generic translation model may mistranslate drug brand names or confuse "before meals" with "after meals".
`TranslationModel` enforces:
1. **Phonetic Brand-Name Preservation**: Preserves generic and trade names (e.g., Paracetamol $\rightarrow$ పారాసిటమాల్, Dolo $\rightarrow$ డోలో, Metformin $\rightarrow$ మెట్ఫార్మిన్).
2. **Deterministic Dosage Directions**: Directly anchors crucial phrases ("after food", "before meals", "twice daily", "at bedtime") to verified trilingual medical dictionaries.
3. **Multi-Way Support**: Direct translation between English, Telugu, and Hindi.

---

## 2. Directory Structure

```text
TranslationModel/
├── src/smartvision_translation/
│   ├── __init__.py           # Package exports (MedicalTranslator, get_translator)
│   ├── translator.py         # Domain-guided translation engine
│   └── dictionary.py         # Curated trilingual pharmaceutical dictionary
├── scripts/
│   └── translate_instruction.py # CLI demonstration tool
├── tests/
│   └── test_translation.py  # Unit test suite
├── requirements.txt
└── README.md
```

---

## 3. Quick Start

### Installation
```bash
cd TranslationModel
pip install -r requirements.txt
```

### CLI Demo
```bash
python scripts/translate_instruction.py
# Or custom input:
python scripts/translate_instruction.py -t "Take 1 tablet after food" -s en -r te
```

### Python API
```python
from smartvision_translation import get_translator

translator = get_translator()
result = translator.translate("Take 1 tablet twice daily after food", source_lang="en", target_lang="te")
print(result.translated_text)
# Output: "1 టాబ్లెట్ రోజుకు రెండుసార్లు భోజనం తర్వాత"
```
