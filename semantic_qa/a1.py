def ngram_overlap(keyword, text, n=3):
    keyword = keyword.lower()
    text = text.lower()
    if keyword in text:
        return 1.0
    grams = [keyword[i:i+n] for i in range(len(keyword)-n+1)] if len(keyword) >= n else [keyword]
    overlap = sum(g in text for g in grams)
    return overlap / len(grams) if grams else 0.0

# 用法
keyword = "*MS*-i30061-SD方向盘控制学习"
text1 = "*MS*-i30061-SD方向盘控制学习"
text2 = "*XT*-i30061-SD方向盘控制学习"

print(ngram_overlap(keyword, text1, n=3))  # 分数较低
print(ngram_overlap(keyword, text2, n=3))  # 分数为1.0