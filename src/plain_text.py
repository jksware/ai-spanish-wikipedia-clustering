#
#  Wikipedia Clustering
#  Copyright (C) 2015 Juan Carlos Pujol Mainegra, Damian Valdés Santiago
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.  
#

from __future__ import print_function
# -*- coding:utf-8 -*-
__author__ = 'Juanca'

import sqlite3
from time import time
import re
from functools import reduce
from textwrap import wrap, fill

from xml_tree import TEMPLATE_NAME, RemoveTags, PrintXML
from xml.etree import ElementTree
from xml.etree.ElementTree import XMLParser


def construct_regex_pipeline():
    regex_first_section_only = re.compile(r'==.*')
    regex_first_section_only_repl = r''

    regex_open_keys = re.compile(r'\{\{')
    regex_open_keys_repl = r'<' + TEMPLATE_NAME + '>'

    regex_close_keys = re.compile(r'}}')
    regex_close_keys_repl = r'</' + TEMPLATE_NAME + '>'

    regex_brackets = re.compile(r'<(?:.(?!>))+.?>')
    regex_brackets_repl = r''

    regex_internal_link = re.compile(r'\[\[(?P<link>(?:.(?!]]))*.)]]')
    regex_internal_link_repl = r'\g<link>'

    regex_external_links = re.compile(r"\[(?!\[)(?:.(?!]))*.]")
    regex_external_links_repl = r''

    regex_tables = re.compile(r'\{\|(?:.(?!\|}))*.\|}')
    regex_tables_repl = r''

    regex_sections = re.compile(r'==(?:.(?!==))*.==')
    regex_sections_repl = r''

    regex_junk_left = re.compile(r'[^áéíóúñäëïöüÁÉÍÓÚÑÄËÏÖÜa-zA-Z\{}]|\b.{1,2}\b')
    regex_junk_left_repl = r' '

    regex_spaces = re.compile(r'\s+')
    regex_spaces_repl = r' '

    regex_pipeline = [
        ('first section only', regex_first_section_only, regex_first_section_only_repl),
        ('brackets', regex_brackets, regex_brackets_repl),
        ('internal links', regex_internal_link, regex_internal_link_repl),
        ('external links', regex_external_links, regex_external_links_repl),
        ('tables', regex_tables, regex_tables_repl),
        ('sections', regex_sections, regex_sections_repl),
        ('junk left', regex_junk_left, regex_junk_left_repl),
        ('open keys', regex_open_keys, regex_open_keys_repl),
        ('close keys', regex_close_keys, regex_close_keys_repl),
        ('spaces', regex_spaces, regex_spaces_repl)
    ]

    return regex_pipeline


def plain_text(wiki_path=None, miniwiki_path=None, verbose=False):
    if not wiki_path:
        wiki_path = 'wiki.db'

    if not miniwiki_path:
        miniwiki_path = 'miniwiki.db'

    print("Processing into plaintext...")
    begin_time = time()

    connection = sqlite3.connect(wiki_path)
    connection.enable_load_extension(True)
    connection.row_factory = sqlite3.Row
    connection.text_factory = str

    connection_mini = sqlite3.connect(miniwiki_path)
    connection_mini.enable_load_extension(True)
    connection_mini.row_factory = sqlite3.Row
    connection_mini.text_factory = str

    cursor_mini = connection_mini.cursor()

    cursor_mini.execute("DROP TABLE IF EXISTS topics")
    cursor_mini.execute("DROP TABLE IF EXISTS corpus_mini")

    cursor_mini.execute("""
    CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY,
        desc TEXT NOT NULL
    )
    """)

    cursor_mini.execute("""
    CREATE TABLE IF NOT EXISTS corpus_mini (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL ,
        user INTEGER NOT NULL,
        timestamp DATETIME not NULL,
        topic INTEGER NOT NULL
            REFERENCES topics (id),
        text TEXT NOT NULL
    )
    """)

    cursor = connection.cursor()

    cursor.execute("SELECT * FROM corpus WHERE text LIKE '%Ficha %'")

    with open('topics_min.txt', encoding='utf-8') as topics_file:
        topics = topics_file.readlines()

    topics = [topic[:-1] if topic[-1] == '\n' else topic for topic in topics]
    topics_dict = {value: i for i, value in enumerate(topics)}

    cursor_mini.executemany("INSERT INTO topics (id, desc) VALUES (?, ?)", enumerate(topics))
    connection_mini.commit()

    topics_str = reduce(lambda x, y: x + '|' + y, topics)
    regex_search = re.compile('({{ *' + topics_str + ')')

    if verbose:
        print(topics_str)

    processed_count = 0
    error_count = 0
    topics = {}

    def stats():
        end_time = time()
        print()
        print('Processed:', processed_count)
        print('Error:', error_count)
        print('Percent:', (processed_count - error_count) * 100 / processed_count)
        print('Elapsed time', (end_time - begin_time))

    regex_pipeline = construct_regex_pipeline()

    for row in cursor:
        title = row['title']
        corpus = str(row['text'])
        if verbose:
            print(title)

        match = regex_search.search(corpus)
        if not match:
            if verbose:
                print("Match template: No match found! Check for encoding issues.")
            continue

        processed_count += 1
        if processed_count % 10000 == 0:
            stats()
            connection_mini.commit()

        topic = match.group(0)
        topics[title] = topic
        if verbose:
            print('Topic', topic)

        corpus_preprocess = corpus
        for regex_step_name, regex_step, regex_step_repl in regex_pipeline:
            if verbose:
                print(regex_step_name)
                print(regex_step.findall(corpus_preprocess))
            corpus_preprocess, _ = regex_step.subn(regex_step_repl, corpus_preprocess)

        # FOR XMLing
        corpus_preprocess = '<data>' + corpus_preprocess + '</data>'

        try:
            target = RemoveTags()
            parser = XMLParser(target=target, encoding='utf-8')
            parser.feed(corpus_preprocess)
            corpus_preprocess = parser.close()

        except ElementTree.ParseError:
            error_count += 1
            continue

        if verbose:
            print()
            print("CORPUS")
            print(fill(corpus, width=80))

            print()
            print("CLEAN")
            print(fill(corpus_preprocess, width=80))

        try:
            new_row = (title, row['user'], row['timestamp'], topics_dict[topic], corpus_preprocess)
            cursor_mini.execute("INSERT INTO corpus_mini (title, user, timestamp, topic, text) VALUES (?, ?, ?, ?, ?)",
                                new_row)
        except KeyError as key_error:
            print(key_error)
            print(new_row)

    connection_mini.commit()
    connection_mini.close()

    connection.close()
    stats()

plain_text('c:\wiki\wiki.db', 'c:\wiki\miniwiki.db', verbose=False)
