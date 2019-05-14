import pandas as pd

suits = ['H','S','C','D']

card_val = (list(range(1,11)) + [10]*3)*4

base_names = ['A'] +list(range(2,11)) + ['J','Q','K']

cards = []

for suit in suits :
    cards.extend(str(num) + suit  for num in base_names)

deck = pd.Series(card_val ,index = cards)

def draw(deck, n = 5):
    return deck.sample(n)

draw(deck)

get_suit = lambda cards : cards[-1]

deck.groupby(get_suit, group_keys = False).apply(draw, n=2)

