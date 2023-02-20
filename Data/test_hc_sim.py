# pipeline: data science
# project: bib-scripts

import pandas as pd
import logging
import sys

sys.path.append("../")
from HCsim import HCsim

def main():
    text1 = """
    The air was thick with the pungent odor of coal smoke as I made my way down the crowded street,
    the throngs of people jostling and bustling past one another in a flurry of activity. The sky
    above was obscured by a murky haze that seemed to weigh heavily on the spirits of all who walked
    beneath it. Despite the bleakness of the scene, there was a sense of vitality in the air, an
    energy that pulsed through the veins of the city like a beating heart. It was a place of
    contradictions, a world in which poverty and wealth, hope and despair, were all intermingled
    in a tumultuous dance. And as I walked, I couldn't help but wonder what fate had in store for me
    in this place, where the past and the future seemed to collide in a dizzying, ever-shifting present.
    """

    text2 = """In the heart of London, amidst the bustling throngs of people, there lived a poor little
    boy named Timmy. He was but a mere waif, with tattered clothes and a hungry stomach, wandering the
    streets in search of a morsel to eat. Despite his plight, he remained optimistic, for he had heard 
    of a kind and generous man who lived in the grand house on the hill. This man, it was said, would take
    in lost children and give them a warm bed to sleep in, a hearty meal to eat, and a chance to learn and
    grow. And so, with a glimmer of hope in his eyes, Timmy set off on a journey towards the house on the
    hill, braving the dangers and hardships of the road, in the hope that he too might find a place to call
    home.
    """

    print("HC-similarity = ", HCsim(max_features=250)(text1, text2))

if __name__ == '__main__':
    main()
