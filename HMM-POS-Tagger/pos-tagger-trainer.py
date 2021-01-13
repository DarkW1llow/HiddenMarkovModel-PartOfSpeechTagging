import pickle

# Khởi tạo dữ liệu cho mạng HMM.
training_file = open("training/hmm-training.txt", "r")
all_lines = training_file.readlines()
training_file.close()

list_of_tags = []
tag_bigram = {}
tag_word_likelihood = {}

# Lấy dữ liệu từ tập training rồi truyền vào mảng
prev_tag = "<S>"
for line in all_lines:
    if prev_tag == "<E>":
        prev_tag = "<S>"
        continue

    word, tag = line.split()

    if tag == ".":
        tag = "<E>"

    if tag not in list_of_tags:
        list_of_tags.append(tag)

    if prev_tag not in tag_bigram:
        tag_bigram[prev_tag] = {}
    if tag not in tag_bigram[prev_tag]:
        tag_bigram[prev_tag][tag] = 1
    tag_bigram[prev_tag][tag] += 1

    prev_tag = tag
    if tag == "<E>":
        continue

    if tag not in tag_word_likelihood:
        tag_word_likelihood[tag] = {}
    if word not in tag_word_likelihood[tag]:
        tag_word_likelihood[tag][word] = 1
    tag_word_likelihood[tag][word] += 1

# Tính toán xác suất để thêm thẻ bigram
for tag in tag_bigram:
    for next_tag in list_of_tags:
        if next_tag not in tag_bigram[tag]:
            tag_bigram[tag][next_tag] = 1
    tag_count = 0
    for next_tag in tag_bigram[tag]:
        tag_count += tag_bigram[tag][next_tag]
    for next_tag in tag_bigram[tag]:
        tag_bigram[tag][next_tag] /= tag_count

# Tính toán xác suất để gán nhãn cho từ mới (không có trong training-set)
for tag in list_of_tags:
    if tag not in tag_word_likelihood:
        tag_word_likelihood[tag] = {}
for tag in tag_word_likelihood:
    tag_count = 0
    tag_word_likelihood[tag]["<U>"] = 1
    unk_words = []
    for word in tag_word_likelihood[tag]:
        tag_count += tag_word_likelihood[tag][word]
        if tag_word_likelihood[tag][word] == 2 and word != "<U>":
            unk_words.append(word)
            tag_word_likelihood[tag]["<U>"] += 1
    for word in unk_words:
        tag_word_likelihood[tag].pop(word)
    for word in tag_word_likelihood[tag]:
        tag_word_likelihood[tag][word] /= tag_count

list_of_tags.remove("<E>")

model_file = open("training/model.txt", "wb")
model = [list_of_tags, tag_bigram, tag_word_likelihood]

pickle.dump(model, model_file)
model_file.close()
