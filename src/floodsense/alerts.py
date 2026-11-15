"""Community-friendly early-warning messages in Bangla and English."""

from __future__ import annotations

from .config import DISTRICTS

_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

DISCLAIMER_BN = (
    "এটি একটি সিদ্ধান্ত-সহায়ক পূর্বাভাস। সরকারি বন্যা পূর্বাভাস ও সতর্কীকরণ কেন্দ্রের (FFWC) "
    "সতর্কবার্তা অনুসরণ করুন। জরুরি সহায়তার জন্য ৯৯৯ নম্বরে ফোন করুন।"
)
DISCLAIMER_EN = (
    "This is a decision-support forecast. Follow official FFWC warnings. "
    "For emergency help, call 999."
)


def to_bangla_digits(value: int | float | str) -> str:
    return str(value).translate(_BN_DIGITS)


def build_alert_bn(district_key: str, level: str, horizon_hours: int = 48) -> str:
    name = DISTRICTS[district_key].name_bn
    h = to_bangla_digits(horizon_hours)
    if level == "High":
        body = (
            f"{name}: সতর্কতা! আগামী {h} ঘণ্টায় বন্যার ঝুঁকি বেশি। গবাদিপশু, শুকনো খাবার, "
            "বিশুদ্ধ পানি, ওষুধ ও জরুরি কাগজপত্র নিরাপদ উঁচু স্থানে সরিয়ে নিন। "
            "স্থানীয় প্রশাসনের নির্দেশনা মেনে চলুন।"
        )
    elif level == "Medium":
        body = (
            f"{name}: আগামী {h} ঘণ্টায় বন্যার ঝুঁকি মাঝারি। নদীর পানির স্তরের দিকে নজর রাখুন "
            "এবং জরুরি জিনিসপত্র প্রস্তুত রাখুন।"
        )
    elif level == "Low":
        body = f"{name}: আগামী {h} ঘণ্টায় বন্যার ঝুঁকি কম। তবে আবহাওয়ার খবর নিয়মিত শুনুন।"
    else:
        raise ValueError(f"Unknown risk level: {level}")
    return f"{body}\n{DISCLAIMER_BN}"


def build_alert_en(district_key: str, level: str, horizon_hours: int = 48) -> str:
    name = DISTRICTS[district_key].name_en
    if level == "High":
        body = (
            f"{name}: WARNING. High flood risk in the next {horizon_hours} hours. Move livestock, "
            "dry food, clean water, medicine and documents to safe high ground and follow local authorities."
        )
    elif level == "Medium":
        body = (
            f"{name}: Moderate flood risk in the next {horizon_hours} hours. Watch river levels "
            "and keep emergency supplies ready."
        )
    elif level == "Low":
        body = f"{name}: Low flood risk in the next {horizon_hours} hours. Keep following weather updates."
    else:
        raise ValueError(f"Unknown risk level: {level}")
    return f"{body}\n{DISCLAIMER_EN}"
