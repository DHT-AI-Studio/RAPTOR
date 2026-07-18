# Standalone test for Fix 1B date extraction. No LLM / Neo4j / deps needed.
# Run from the app/ dir:   python3 test_date_utils.py

from date_utils import extract_first_date, fill_relation_dates


def check(name, got, want):
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def main():
    results = []

    # --- extract_first_date ---
    results.append(check("full date",      extract_first_date("1987 年 2 月 21 日正式成立臺積電"), "1987-02-21"))
    results.append(check("year+month",     extract_first_date("2017年3月臺積電市值超越英特爾"),    "2017-03-01"))
    results.append(check("year only",      extract_first_date("1995年聯電轉型為純晶圓代工廠"),      "1995-01-01"))
    results.append(check("minguo",         extract_first_date("民國76年成立"),                      "1987-01-01"))
    results.append(check("no year (cap)",  extract_first_date("資本額僅 13.775 億元"),              None))
    results.append(check("anniversary 36", extract_first_date("正式成立滿 36 週年"),                None))

    # --- fill_relation_dates: the real TSMC Clip1 summary ---
    summary = (
        "2023 年 2 月 21 日，全球晶圓代工龍頭臺積電正式成立滿 36 週年。"
        "1985 年張忠謀從德州儀器回臺擔任工研院院長。"
        "1987 年 2 月 21 日，工研院與飛利浦合資於新竹科學園區成立臺積電，由張忠謀擔任董事長兼執行長。"
    )
    entities = [
        {"id": "taiji_dian", "name": "臺積電"},
        {"id": "philips", "name": "飛利浦"},
        {"id": "zhang", "name": "張忠謀"},
    ]
    relations = [
        {"subject_id": "philips", "predicate": "INVESTS_IN", "object_id": "taiji_dian"},
        {"subject_id": "zhang", "predicate": "LEADS", "object_id": "taiji_dian"},
    ]
    n = fill_relation_dates(summary, entities, relations)
    results.append(check("fill count", n, 2))
    results.append(check("philips INVESTS_IN @", relations[0].get("time_start"), "1987-02-21"))
    results.append(check("zhang LEADS @",        relations[1].get("time_start"), "1987-02-21"))

    # already-dated relation must be left untouched
    pre = [{"subject_id": "zhang", "predicate": "LEADS", "object_id": "taiji_dian", "time_start": "1990-01-01"}]
    fill_relation_dates(summary, entities, pre)
    results.append(check("keep existing date", pre[0]["time_start"], "1990-01-01"))

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    print(f"{results.count(False)} / {len(results)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
