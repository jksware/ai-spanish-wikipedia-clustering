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


TEMPLATE_NAME = 'template'


class RemoveTags:
    depth = 0
    final = ''

    def start(self, tag, attrib):   # Called for each opening tag.
        if tag == TEMPLATE_NAME:
            self.depth += 1

    def end(self, tag):             # Called for each closing tag.
        if tag == TEMPLATE_NAME:
            self.depth -= 1

    def data(self, data):
        if self.depth == 0:
            self.final += ' ' + data

    def close(self):    # Called when all data has been parsed.
        return self.final


class PrintXML:
    depth = 0
    final = ''

    def start(self, tag, attrib):   # Called for each opening tag.
        text = '\t' * self.depth + '<{}>'.format(tag)
        self.final += text + '\n'
        self.depth += 1
        print(text)

    def end(self, tag):             # Called for each closing tag.
        self.depth -= 1
        text = '\t' * self.depth + '</{}>'.format(tag)
        self.final += text + '\n'
        print(text)

    def data(self, data):
        text = '\t' * self.depth + data[0:100]
        self.final += text + '\n'
        print(text)

    def close(self):    # Called when all data has been parsed.
        return self.final