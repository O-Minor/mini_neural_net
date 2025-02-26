# goal of file: show how another programmer can use my set of functions
# to feed, train, and then generate from a mini neural net
print("before importing")
import neural_net_functions as nn
print("end importing")
words = []
print("starting adding to words dataset\n")
words = nn.add_col_to_words('pokemon.csv', "Name", words)
print(words[-8:])
words = nn.add_txt_to_words('blorbo2.txt', words)
print(words[-8:])
words = nn.add_txt_to_words('frank_text_clean.txt', words)
print(words[-8:])
words = nn.add_txt_to_words('ai_names_real_fic.txt', words)
print(words[-8:])

print("starting training the neural net\n")
imported = nn.train_ai(words, 3)
print("start generating")
nn.do_generating(imported, "", 10)