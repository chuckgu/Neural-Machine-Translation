#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from konlpy.tag import Twitter
from konlpy.utils import pprint


def korean_morph(text):
    twitter = Twitter()
    
    s=twitter.morphs(str(unicode(text)))
    
    s=' '.join(s)
    
    
    
    return s


