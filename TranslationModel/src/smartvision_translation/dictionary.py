"""Trilingual Medical and Dosage Instruction Dictionary.

Maps critical medication instructions between English, Telugu, and Hindi
to prevent medical hallucination during translation.
"""

from __future__ import annotations

from typing import Dict

PHRASE_DICTIONARY: Dict[str, Dict[str, str]] = {
    # Dosage timing
    "after food": {"en": "after food", "te": "భోజనం తర్వాత", "hi": "भोजन के बाद"},
    "before food": {"en": "before food", "te": "భోజనానికి ముందు", "hi": "भोजन से पहले"},
    "after meals": {"en": "after meals", "te": "భోజనం తర్వాత", "hi": "भोजन के बाद"},
    "twice daily": {"en": "twice daily", "te": "రోజుకు రెండుసార్లు", "hi": "दिन में दो बार"},
    "once daily": {"en": "once daily", "te": "రోజుకు ఒకసారి", "hi": "दिन में एक बार"},
    "thrice daily": {"en": "thrice daily", "te": "రోజుకు మూడుసార్లు", "hi": "दिन में तीन बार"},
    "at bedtime": {"en": "at bedtime", "te": "పడుకునే సమయంలో", "hi": "सोते समय"},
    "in the morning": {"en": "in the morning", "te": "ఉదయం వేళ", "hi": "सुबह के समय"},
    "in the evening": {"en": "in the evening", "te": "సాయంత్రం వేళ", "hi": "शाम के समय"},

    # Dosage form
    "1 tablet": {"en": "1 tablet", "te": "1 టాబ్లెట్", "hi": "1 गोली"},
    "2 tablets": {"en": "2 tablets", "te": "2 టాబ్లెట్లు", "hi": "2 गोलियां"},
    "take 1 tablet": {"en": "take 1 tablet", "te": "1 టాబ్లెట్ తీసుకోండి", "hi": "1 गोली लें"},
    "take with water": {"en": "take with water", "te": "నీటితో తీసుకోండి", "hi": "पानी के साथ लें"},
    "with warm water": {"en": "with warm water", "te": "గోరువెచ్చని నీటితో", "hi": "गुनगुने पानी के साथ"},

    # Warnings and Storage
    "store in a cool dry place": {
        "en": "store in a cool dry place",
        "te": "చల్లని పొడి ప్రదేశంలో నిల్వ చేయండి",
        "hi": "ठंडी और सूखी जगह पर रखें",
    },
    "protect from light": {
        "en": "protect from light",
        "te": "వెలుతురు తగలకుండా భద్రపరచండి",
        "hi": "रोशनी से बचाएं",
    },
    "keep out of reach of children": {
        "en": "keep out of reach of children",
        "te": "పిల్లలకు అందకుండా ఉంచండి",
        "hi": "बच्चों की पहुंच से दूर रखें",
    },
    "for external use only": {
        "en": "for external use only",
        "te": "బాహ్య వినియోగానికి మాత్రమే",
        "hi": "केवल बाहरी उपयोग के लिए",
    },

    # Common Indications
    "for fever": {"en": "for fever", "te": "జ్వరం కోసం", "hi": "बुखार के लिए"},
    "for pain relief": {"en": "for pain relief", "te": "నొప్పి నివారణ కోసం", "hi": "दर्द से राहत के लिए"},
    "for cold and cough": {"en": "for cold and cough", "te": "జలుబు మరియు దగ్గు కోసం", "hi": "सर्दी और खांसी के लिए"},
}

PHARMACEUTICAL_NAMES: Dict[str, Dict[str, str]] = {
    "paracetamol": {"en": "Paracetamol", "te": "పారాసిటమాల్", "hi": "पैरासिटामोल"},
    "dolo": {"en": "Dolo", "te": "డోలో", "hi": "डोलो"},
    "cetirizine": {"en": "Cetirizine", "te": "సెటిరిజిన్", "hi": "सिटिरिज़िन"},
    "metformin": {"en": "Metformin", "te": "మెట్ఫార్మిన్", "hi": "मेटफॉर्मिन"},
    "amoxicillin": {"en": "Amoxicillin", "te": "అమోక్సిసిలిన్", "hi": "एमोक्सिसिलिन"},
    "pantoprazole": {"en": "Pantoprazole", "te": "పాంటోప్రజోల్", "hi": "पैंटोप्रैजोल"},
    "atorvastatin": {"en": "Atorvastatin", "te": "అటోర్వాస్టాటిన్", "hi": "एटोरवास्टेटिन"},
    "azithromycin": {"en": "Azithromycin", "te": "అజిత్రోమైసిన్", "hi": "एज़िथ्रोमाइसिन"},
}
