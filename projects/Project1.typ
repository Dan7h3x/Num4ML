#let title = [
  Project of Group 1
]
#set page(paper: "a4", margin: (x: 1.6cm, y: 1cm), numbering: "(1/1)")
#let Colored(term, color: purple) = { emph(text(color)[#term]) }
#import "config.typ": *

#align(center, text(22pt)[*#title*])

= Description

#figure(image("DataSets.png", width: 80%), outlined: true)

First apply a classic classification #emph(text(purple)[(the choise of methodology is arbitrary)]),
Then apply one of #emph(text(green.darken(50%))[HOSVD]) or #Colored([HOOI], color: green.darken(50%)) to
improve the #Colored([accuracy], color: blue) and other performance metrics of
the classification method. For description of HOOI and HOSVD, please read #underline[General Intro.pdf].

- #underline[ You must have a comparison based on the increase of ranks. ]\
- #underline[ You must report the time of computations in case of speed comparison. ]

$bold("Hint:")$ use #emph(
  text(size: 15pt, fill: gradient.linear(..color.map.rainbow))[Colab Jupyter],
), since the computations maybe take some time and RAM.
