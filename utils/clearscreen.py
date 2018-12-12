import os


def clearscreen():
    # os.system('cls' if os.name == 'nt' else 'clear')
    print('\033[H\033[J')
