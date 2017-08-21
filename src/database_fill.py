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

import os
import glob
import json
import sqlite3


def fill_database(db_path=None, json_path=None):
    # connection = sqlite3.connect(':memory:')
    if not db_path:
        db_path = 'wiki.db'

    if not json_path:
        json_path = ''

    connection = sqlite3.connect(db_path)
    connection.enable_load_extension(True)

    cursor = connection.cursor()

    # cursor.execute("DROP TABLE IF EXISTS usernames")
    # cursor.execute("DROP TABLE IF EXISTS corpus")
    # cursor.execute("DROP TABLE IF EXISTS redirects")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usernames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS corpus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL ,
        user INTEGER NOT NULL
            REFERENCES user (id),
        timestamp DATETIME not NULL,
        text TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS redirects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        user INTEGER NOT NULL
            REFERENCES user (id),
        timestamp DATETIME NOT NULL,
        text TEXT NOT NULL
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS corpus_index ON corpus (title)")
    cursor.execute("CREATE INDEX IF NOT EXISTS redirects_index ON redirects (title)")

    docs = glob.glob(os.path.join(json_path, '*.json'))

    doc_count = len(docs)
    last_user_id = 0
    all_users_set = set()
    all_users_dict = {}

    incomplete_sql_command = "INSERT INTO {} (title, user, timestamp, text) VALUES (?, ?, ?, ?)"
    corpus_insert_sql_command = incomplete_sql_command.format('corpus')
    redirects_insert_sql_command = incomplete_sql_command.format('redirects')

    print('Will process {} documents.'.format(doc_count))

    for i, doc in enumerate(docs):
        print("Processing {} out of {} documents: {}".format(i + 1, doc_count, doc))

        with open(doc) as json_doc:
            lines = json_doc.readlines()

        json_lines = [json.loads(line) for line in lines]

        file_users = set(line['username'] if 'username' in line else line['ip'] for line in json_lines)

        new_users_set = file_users.difference(all_users_set)
        new_users_dict = {user: last_user_id + i for i, user in enumerate(new_users_set)}
        all_users_set.update(new_users_set)
        all_users_dict.update(new_users_dict)

        last_user_id += len(new_users_set)

        cursor.executemany("INSERT INTO usernames (username, id) VALUES (?, ?)", new_users_dict.items())

        cursor.executemany(corpus_insert_sql_command,
                           ((info['title'],
                             all_users_dict[info['username'] if 'username' in info else info['ip']],
                             info['timestamp'],
                             info['text'])
                            for info in json_lines if 'redirect' not in info))

        cursor.executemany(redirects_insert_sql_command,
                           ((info['title'],
                             all_users_dict[info['username'] if 'username' in info else info['ip']],
                             info['timestamp'],
                             info['text'])
                            for info in json_lines if 'redirect' in info))

    connection.commit()
    connection.close()

fill_database(r'c:\wiki\wiki.db', r'c:\wiki\articles')