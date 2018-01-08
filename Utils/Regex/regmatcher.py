#!/usr/bin/env python
# -*- coding: utf-8 -*-

import regex as re
import pandas as pd
import os

class RegexMatcher:
    def __init__(self,debug=False):
        self.initialize_regex()
        self.debug = debug


    def load_negative_name(self):
        neg_names = []
        for line in open('./Utils/Regex/negative_name.csv', 'r'):
            neg_names.append(line.strip())

        r_neg_name = r'(\d+|' + '|'.join(neg_names) + r')'
        return r_neg_name

    def initialize_regex(self):
        "---- Regular Expression for Baby Name Detect ---"
        def opt_r(r):
            return r'(?:(' + r + r')?)'

        def in_bracket(r):
            return r'(' + r + r')'

        r_neg_names = self.load_negative_name()

        r_danhxung = r'((?<=\s|^)((C|c)hị|(G|g)ái|(T|t)rai|(C|c)on_(g|G)ái|(C|c)on_(t|T)rai|(A|a)nh_(T|t)rai|(E|e)m_(g|G)ái|(E|e)m_(t|T)rai|(B|b)ạn|' \
                     r'(E|e)m|(C|c)háu|(B|b)é|(C|c)on|(C|c)u|(c|C)ục[_\s](c|C)ưng|(Đ|đ)ại[_\s](K|C|k|c)a))'
        r_sohuu = r'((c|C)ủa)'
        r_dacdiem = r'((c|C)ưng|(N|g)oan|(y|Y)êu|(t|T)húi|(Ơ|ơ)i|(K|k|c|C)ute|((M|m)au|(H|h)ay)\s(Ă|ă)n\s(c|C)hóng\s(l|L)ớn|(S|s)nvv|(S|s)inh_(N|n)hật[\s_](V|v)ui_(V|v)ẻ)'
        r_quanhe = r'(((B|b)(ố|a)|(M|m)(ẹ|á)|(A|a)nh|(C|c)hị|(C|c)hú|(B|b)ác|(C|c)ô|(D|d)ì|(Ô|ô)ng|(B|b)à)' \
                   r'((\s|_)((B|b)ố|(M|m)ẹ|(A|a)nh|(C|c)hị|(C|c)hú|(B|b)ác|(C|c)ô|(D|d)ì|(Ô|ô)ng|(B|b)à))?)'
        r_ngaythang = r'((H|h)ôm_nay|(H|h)ôm_qua|(H|h)ôm_kia|(N|n)gày_mai|(N|n)gày_kia)'

        r_ngaydacbiet = r'((đ|Đ)ầy_(T|t)háng|(b|B)irthday|(t|T)hôi_(N|n)ôi|(s|S)inh_(N|n)hật|(M|m)ừng)'
        r_prefixtuoi = r'(((Đ|đ)ã|(V|v)ừa|(S|s)ắp)?(_|\s)?(T|t)ròn)'

        r_s = r'\s'
        r_or = r'|'
        """
        r_baby_name = ur'(((?<=\s|^)(?!(' + r_ngaydacbiet + r_or + r_sohuu + r_or + r_danhxung + r_or + r_dacdiem + r_or + r_quanhe \
                      + r_or + r_prefixtuoi + r_or + r_ngaythang + r_or + r_neg_names + ur')\s)\w+\s(([-&]|và)\s)?){1,4}(\p{Lu}[\p{Ll}_]+\s)*)'

        r_a = in_bracket(in_bracket(r_ngaydacbiet + r_or + r_danhxung) + r_s)
        r_b = in_bracket(
            in_bracket(r_dacdiem + r_or + r_prefixtuoi + r_or + in_bracket(r_sohuu + r_s + r_quanhe)) + r_s)

        r_baby_left = in_bracket(r_a + in_bracket(
            opt_r(r_ngaydacbiet + r_s) + opt_r(r_sohuu + r_s) + opt_r(r_danhxung + r_s + opt_r(r_dacdiem + r_s))))
        r_baby_right = in_bracket(
            r_b + in_bracket(
                opt_r(r_dacdiem + r_s) + opt_r(r_sohuu + r_s) + opt_r(r_quanhe + r_s) + opt_r(r_prefixtuoi + r_s)))

        r_baby = ur'((' + r_baby_left + r_baby_name + r_baby_right + ur')|' \
                 + ur'(' + r_baby_left + r_baby_name + ur')|' \
                 + ur'(' + r_baby_name + r_baby_right + ur')|' \
                 + ur'((\(|\"\")\s((\p{Lu}[\p{Ll}_]+(\s([-&]\s)?\p{Lu}[\p{Ll}_]+){0,3}))\s(\)|\"\"))' + ur')'

        """
        r_baby_name = r'(((?<=\s|^)(?!(r_ngaydacbiet|r_sohuu|r_danhxung|r_dacdiem|r_quanhe|r_prefixtuoi|r_ngaythang|r_neg_names)\s)\w+\s(([-&]|và)\s)?){1,4}(\p{Lu}[\p{Ll}_]+\s)*)'
        r_a = r'((r_ngaydacbiet|r_danhxung)r_s)'
        r_b = r'((r_dacdiem|r_prefixtuoi|(r_sohuur_sr_quanhe))r_s)'

        r_baby_left = r'(r_a((?:(r_ngaydacbietr_s)?)(?:(r_sohuur_s)?)(?:(r_danhxungr_s(?:(r_dacdiemr_s)?))?)))'
        r_baby_right = r'(r_b((?:(r_dacdiemr_s)?)(?:(r_sohuur_s)?)(?:(r_quanher_s)?)(?:(r_prefixtuoir_s)?)))'
        r_baby = r'((r_baby_leftr_baby_namer_baby_right)|(r_baby_leftr_baby_name)|(r_baby_namer_baby_right)|' \
            + r'((\(|\"\")\s((\p{Lu}[\p{Ll}_]+(\s([-&]\s)?\p{Lu}[\p{Ll}_]+){0,3}))\s(\)|\"\")))'

        r_baby = re.sub("r_baby_left", r_baby_left, r_baby)
        r_baby = re.sub("r_baby_right", r_baby_right, r_baby)
        r_baby = re.sub("r_baby_name", r_baby_name, r_baby)
        r_baby = re.sub("r_a", r_a, r_baby)
        r_baby = re.sub("r_b", r_b, r_baby)
        r_baby = re.sub("r_prefixtuoi", r_prefixtuoi, r_baby)
        r_baby = re.sub("r_ngaydacbiet", r_ngaydacbiet, r_baby)
        r_baby = re.sub("r_ngaythang", r_ngaythang, r_baby)
        r_baby = re.sub("r_quanhe", r_quanhe, r_baby)
        r_baby = re.sub("r_dacdiem", r_dacdiem, r_baby)
        r_baby = re.sub("r_sohuu", r_sohuu, r_baby)
        r_baby = re.sub("r_danhxung", r_danhxung, r_baby)
        r_baby = re.sub("r_neg_names", r_neg_names, r_baby)
        r_baby = re.sub("r_s", r_s, r_baby)

        self.name_reg = re.compile(r_baby, flags=re.UNICODE)

        "---- End Regular Expression for Baby Name Detect ---"

        r_ngaydacbiet_age = r'((đ|Đ)ầy_(T|t)háng|(b|B)irthday|(t|T)hôi_(N|n)ôi|(s|S)inh_(N|n)hật|(S|s)nzz|(S|s)nvv)'
        r_tuoi = r"(<number>((_|\s)(T|t)háng)?(_|\s)(T|t)uổi)"
        r_age = r"r_ngaydacbiet_age|r_tuoi"

        r_age = re.sub("r_ngaydacbiet_age", r_ngaydacbiet_age, r_age)
        r_age = re.sub("r_tuoi", r_tuoi, r_age)
        r_age = re.sub("r_prefixtuoi", r_prefixtuoi, r_age)

        self.age_reg = re.compile(r_age, flags=re.UNICODE)
        "---- End Regular Expression for Baby Age Detect ---"

    def calc(self,word_ids):
        sentence = ' '.join(word_ids)
        sentence = ' '.join([word[0] + word[1:].lower() for word in sentence.split(' ')]) + ' '

        matches = re.finditer(self.name_reg, sentence)
        res = ['0' if char != ' ' else ' ' for char in sentence]
        matches_age = re.finditer(self.age_reg, sentence)

        if self.debug:
            print('-----------------------------------------')

        for matchNum, match in enumerate(matches):
            matchNum = matchNum + 1

            if self.debug:
                print(("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                                     end=match.end(),
                                                                                     match=match.group().encode('utf-8'))))

            (s_id, e_id) = (match.start(), match.end())
            res[s_id:e_id] = ['1' if res[c_id] != ' ' else ' ' for c_id in range(s_id, e_id)]

        for matchNum, match in enumerate(matches_age):
            matchNum = matchNum + 1

            if self.debug:
                print(("Age: Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                               end=match.end(),
                                                                               match=match.group().encode('utf-8'))))

            (s_id, e_id) = (match.start(), match.end())
            """res[s_id:e_id] = [('2' if res[c_id] == '0' else '1') if res[c_id] != ' '  else ' ' for c_id in
                              xrange(s_id, e_id)]"""
            res[s_id:e_id] = ['2' if res[c_id] != ' '  else ' ' for c_id in
                              range(s_id, e_id)]

        str_res = (''.join(res)).strip()

        if self.debug:
            print(sentence.strip())
            print(str_res)
            print('-----------------------------------------')

        return str_res


    def annotate_name(self,word_ids):
        reg_sent = self.calc(word_ids)
        return [reg_word[0] for reg_word in reg_sent.split(' ')]

'''
def load_file(path_to_file):
    assert os.path.isfile(path_to_file)

    df = pd.read_csv(path_to_file, encoding='utf-8')
    df.dropna(inplace=True)

    start_post = '['
    end_post = ']'

    sents = []
    labels = []
    tags = []

    cur_sent = []
    cur_label = []
    cur_tag = []

    for index, row in df.iterrows():
        token = row['token']
        label = row['label']
        tag = row['pos']

        if token in [start_post, end_post]:
            if len(cur_sent) > 1:
                sents.append(cur_sent)
                labels.append(cur_label)
                tags.append(cur_tag)
            cur_sent = []
            cur_label = []
            cur_tag = []
        else:
            cur_sent.append(token)
            cur_label.append(label)
            cur_tag.append(tag)

    return (sents, labels, tags)


def flattern(A):
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flattern(i))
        else:
            rt.append(i)
    return rt
'''

if __name__ == '__main__':
    regex_matcher = RegexMatcher()


    word_ids = 'chúc Lam và Trí sinh_nhật vui_vẻ'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'lắc_lư theo điệu nhạc của chị Trang Moon và suy_nghĩ về nước Mỹ'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'đầy_tháng con_nhà kiều Tũn Nguyễn Quoc Hung'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'chúc_mừng cháu yêu Minh Khôi'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'chúc_mừng gia_đình đại ka Jimmy Tào có thêm thiên_thần mới'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = '"" Ku Tom "" ( Phạm Thị Hoa ) ( pham Thi Hong ) ( Pham Ngoc Cam_Tu ) ( Phạm Cẩm_Tú ) ( 14/02/2017 )'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Vì vậy mấy mem đừng thả hoa khi em post hình này lên nhé'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Tròn 2 tuoi em đã biết gọi điện thoại nói_chuyện với mẹ nheo_nhẽo hỏi xem mẹ đỡ chưa ?'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'chúc bé Hary mau ăn chóng lớn . Thảo hương của mẹ tròn 5 tuổi'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = '1 Thảo hương của mẹ tròn 5 tuổi'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Đầy_tháng bé Nhộng ( Phạm Ngọc Cẩm_Tú )'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Đầy_tháng cục cưng Tánh Kì'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Happy Birthday đại_ca Mimi'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Hôm_nay sinh_nhật. Tặng quà cho bé Mimi yêu nè'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Mừng Mimi - Uyên Nhi tròn 2 tuổi'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Qua đầy_tháng Nana'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Hôm_nay Ken yêu của mẹ tròn'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Thôi_nôi cu Beo'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Sinh_nhật cháu Phan Anh'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Sinh_nhật Đăng Khoa'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Sinh_nhật ST Hong Ngu'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Chúc bé My & Mu snvv'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Thôi_nôi tien dat'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Đầy_tháng Tony của mẹ lam'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'chúc bé Hary mau ăn chóng lớn'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Ngu si dốt nát đần'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Đặc_biệt là bớt nghịch chút con yêu nhé'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    word_ids = 'Yêu con'.split(' ')
    print(regex_matcher.annotate_name(word_ids))

    '''

    # MAIN
    sents, labels, tags = load_file('./final_must_be_labelled_excel.csv')
    reg_names = [annotate_name(word_ids) for word_ids in sents]

    df = pd.DataFrame(columns=['label', 'pos', 'token', 'reg_name'])

    df['token'] = flattern([['['] + sent + [']'] for sent in sents])
    df['pos'] = flattern([['N'] + tag + ['N'] for tag in tags])
    df['label'] = flattern([['O'] + label + ['O'] for label in labels])
    df['reg_name'] = flattern([['0'] + reg_name + ['0'] for reg_name in reg_names])

    df.to_csv('tmp2.csv', encoding='utf-8', index=False)
    '''
    pass

