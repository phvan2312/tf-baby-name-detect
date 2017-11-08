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
        for line in open('./negative_name.csv', 'r'):
            neg_names.append(line.decode('utf-8').strip())

        r_neg_name = ur'(\d+|' + '|'.join(neg_names) + ur')'
        return r_neg_name

    def initialize_regex(self):
        "---- Regular Expression for Baby Name Detect ---"
        def opt_r(r):
            return ur'(?:(' + r + ur')?)'

        def in_bracket(r):
            return ur'(' + r + ur')'

        r_neg_names = self.load_negative_name()

        r_danhxung = ur'((?<=\s|^)((C|c)hị|(G|g)ái|(T|t)rai|(C|c)on_(g|G)ái|(C|c)on_(t|T)rai|(A|a)nh_(T|t)rai|(E|e)m_(g|G)ái|(E|e)m_(t|T)rai|(B|b)ạn|' \
                     ur'(E|e)m|(C|c)háu|(B|b)é|(C|c)on|(C|c)u|(c|C)ục[_\s](c|C)ưng|(Đ|đ)ại[_\s](K|C|k|c)a))'
        r_sohuu = ur'((c|C)ủa)'
        r_dacdiem = ur'((c|C)ưng|(N|g)oan|(y|Y)êu|(t|T)húi|(Ơ|ơ)i|(K|k|c|C)ute|((M|m)au|(H|h)ay)\s(Ă|ă)n\s(c|C)hóng\s(l|L)ớn|(S|s)nvv|(S|s)inh_(N|n)hật[\s_](V|v)ui_(V|v)ẻ)'
        r_quanhe = ur'(((B|b)(ố|a)|(M|m)(ẹ|á)|(A|a)nh|(C|c)hị|(C|c)hú|(B|b)ác|(C|c)ô|(D|d)ì|(Ô|ô)ng|(B|b)à)' \
                   ur'((\s|_)((B|b)ố|(M|m)ẹ|(A|a)nh|(C|c)hị|(C|c)hú|(B|b)ác|(C|c)ô|(D|d)ì|(Ô|ô)ng|(B|b)à))?)'
        r_ngaythang = ur'((H|h)ôm_nay|(H|h)ôm_qua|(H|h)ôm_kia|(N|n)gày_mai|(N|n)gày_kia)'

        r_ngaydacbiet = ur'((đ|Đ)ầy_(T|t)háng|(b|B)irthday|(t|T)hôi_(N|n)ôi|(s|S)inh_(N|n)hật|(M|m)ừng)'
        r_prefixtuoi = ur'(((Đ|đ)ã|(V|v)ừa|(S|s)ắp)?(_|\s)?(T|t)ròn)'

        r_s = ur'\s'
        r_or = ur'|'
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
        r_baby_name = ur'(((?<=\s|^)(?!(r_ngaydacbiet|r_sohuu|r_danhxung|r_dacdiem|r_quanhe|r_prefixtuoi|r_ngaythang|r_neg_names)\s)\w+\s(([-&]|và)\s)?){1,4}(\p{Lu}[\p{Ll}_]+\s)*)'
        r_a = ur'((r_ngaydacbiet|r_danhxung)r_s)'
        r_b = ur'((r_dacdiem|r_prefixtuoi|(r_sohuur_sr_quanhe))r_s)'

        r_baby_left = ur'(r_a((?:(r_ngaydacbietr_s)?)(?:(r_sohuur_s)?)(?:(r_danhxungr_s(?:(r_dacdiemr_s)?))?)))'
        r_baby_right = ur'(r_b((?:(r_dacdiemr_s)?)(?:(r_sohuur_s)?)(?:(r_quanher_s)?)(?:(r_prefixtuoir_s)?)))'
        r_baby = ur'((r_baby_leftr_baby_namer_baby_right)|(r_baby_leftr_baby_name)|(r_baby_namer_baby_right)|' \
            + ur'((\(|\"\")\s((\p{Lu}[\p{Ll}_]+(\s([-&]\s)?\p{Lu}[\p{Ll}_]+){0,3}))\s(\)|\"\")))'

        r_baby = re.sub(u"r_baby_left", r_baby_left, r_baby)
        r_baby = re.sub(u"r_baby_right", r_baby_right, r_baby)
        r_baby = re.sub(u"r_baby_name", r_baby_name, r_baby)
        r_baby = re.sub(u"r_a", r_a, r_baby)
        r_baby = re.sub(u"r_b", r_b, r_baby)
        r_baby = re.sub(u"r_prefixtuoi", r_prefixtuoi, r_baby)
        r_baby = re.sub(u"r_ngaydacbiet", r_ngaydacbiet, r_baby)
        r_baby = re.sub(u"r_ngaythang", r_ngaythang, r_baby)
        r_baby = re.sub(u"r_quanhe", r_quanhe, r_baby)
        r_baby = re.sub(u"r_dacdiem", r_dacdiem, r_baby)
        r_baby = re.sub(u"r_sohuu", r_sohuu, r_baby)
        r_baby = re.sub(u"r_danhxung", r_danhxung, r_baby)
        r_baby = re.sub(u"r_neg_names", r_neg_names, r_baby)
        r_baby = re.sub(u"r_s", r_s, r_baby)

        self.name_reg = re.compile(r_baby, flags=re.UNICODE)

        "---- End Regular Expression for Baby Name Detect ---"

        r_ngaydacbiet_age = ur'((đ|Đ)ầy_(T|t)háng|(b|B)irthday|(t|T)hôi_(N|n)ôi|(s|S)inh_(N|n)hật|(S|s)nzz|(S|s)nvv)'
        r_tuoi = ur"(r_prefixtuoi(_|\s)\d{1,2}((_|\s)(T|t)háng)?(_|\s)(T|t)uổi)"
        r_age = ur"r_ngaydacbiet_age|r_tuoi"

        r_age = re.sub(u"r_ngaydacbiet_age", r_ngaydacbiet_age, r_age)
        r_age = re.sub(u"r_tuoi", r_tuoi, r_age)
        r_age = re.sub(u"r_prefixtuoi", r_prefixtuoi, r_age)

        self.age_reg = re.compile(r_age, flags=re.UNICODE)
        "---- End Regular Expression for Baby Age Detect ---"

    def calc(self,word_ids):
        sentence = ' '.join(word_ids)
        sentence = ' '.join([word[0] + word[1:].lower() for word in sentence.split(' ')]) + ' '

        matches = re.finditer(self.name_reg, sentence)
        res = ['0' if char != ' ' else ' ' for char in sentence]
        matches_age = re.finditer(self.age_reg, sentence)

        if self.debug:
            print '-----------------------------------------'

        for matchNum, match in enumerate(matches):
            matchNum = matchNum + 1

            if self.debug:
                print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                                     end=match.end(),
                                                                                     match=match.group().encode('utf-8')))

            (s_id, e_id) = (match.start(), match.end())
            res[s_id:e_id] = ['1' if res[c_id] != ' ' else ' ' for c_id in xrange(s_id, e_id)]

        for matchNum, match in enumerate(matches_age):
            matchNum = matchNum + 1

            if self.debug:
                print ("Age: Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                               end=match.end(),
                                                                               match=match.group().encode('utf-8')))

            (s_id, e_id) = (match.start(), match.end())
            res[s_id:e_id] = [('2' if res[c_id] == '0' else '1') if res[c_id] != ' '  else ' ' for c_id in
                              xrange(s_id, e_id)]

        str_res = (''.join(res)).strip()

        if self.debug:
            print sentence.strip()
            print str_res
            print '-----------------------------------------'

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


    word_ids = u'chúc Lam và Trí sinh_nhật vui_vẻ'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'lắc_lư theo điệu nhạc của chị Trang Moon và suy_nghĩ về nước Mỹ'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'đầy_tháng con_nhà kiều Tũn Nguyễn Quoc Hung'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'chúc_mừng cháu yêu Minh Khôi'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'chúc_mừng gia_đình đại ka Jimmy Tào có thêm thiên_thần mới'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'"" Ku Tom "" ( Phạm Thị Hoa ) ( pham Thi Hong ) ( Pham Ngoc Cam_Tu ) ( Phạm Cẩm_Tú ) ( 14/02/2017 )'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Vì vậy mấy mem đừng thả hoa khi em post hình này lên nhé'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Tròn 2 tuoi em đã biết gọi điện thoại nói_chuyện với mẹ nheo_nhẽo hỏi xem mẹ đỡ chưa ?'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'chúc bé Hary mau ăn chóng lớn . Thảo hương của mẹ tròn 5 tuổi'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'1 Thảo hương của mẹ tròn 5 tuổi'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Đầy_tháng bé Nhộng ( Phạm Ngọc Cẩm_Tú )'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Đầy_tháng cục cưng Tánh Kì'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Happy Birthday đại_ca Mimi'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Hôm_nay sinh_nhật. Tặng quà cho bé Mimi yêu nè'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Mừng Mimi - Uyên Nhi tròn 2 tuổi'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Qua đầy_tháng Nana'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Hôm_nay Ken yêu của mẹ tròn'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Thôi_nôi cu Beo'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Sinh_nhật cháu Phan Anh'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Sinh_nhật Đăng Khoa'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Sinh_nhật ST Hong Ngu'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Chúc bé My & Mu snvv'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Thôi_nôi tien dat'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Đầy_tháng Tony của mẹ lam'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'chúc bé Hary mau ăn chóng lớn'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Ngu si dốt nát đần'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Đặc_biệt là bớt nghịch chút con yêu nhé'.split(' ')
    print regex_matcher.annotate_name(word_ids)

    word_ids = u'Yêu con'.split(' ')
    print regex_matcher.annotate_name(word_ids)

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

