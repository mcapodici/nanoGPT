from convokit import Corpus, download # https://convokit.cornell.edu/documentation/subreddit.html
corpus = Corpus(filename=download("subreddit-sydney"))
textarr = []
for utt in corpus.iter_utterances():
    if utt.text != "[deleted]":
        textarr.append(utt.text)
text = '\n'.join(textarr);
text_file = open("input.txt", "w")
n = text_file.write(text)
text_file.close()