# goal of file: show how another programmer can use my set of functions
# to feed, train, and then generate from a mini neural net
print("before importing")
import neural_net_functions as nn
print("begin dev demo\n")
words = nn.add_col_to_words('pokemon.csv', "Name")
print(words[-8:])

words = nn.add_txt_to_words('ai_names_real_fic.csv', words, ',')
print(words[-8:])

words = nn.add_txt_to_words('blorbo2.txt', words)
print(words[-8:])
print("start training\n")
nn.train_ai(words, 3)
print("end training start gen\n")
nn.do_generating("pika", 10)