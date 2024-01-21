#set text(font: "Noto Serif", size: 12pt, style: "normal")

#set par(justify: true, leading: 0.95em)
// #import "@preview/rose-pine:0.1.0": apply
// #show: apply()
// #show: rest => columns(2,rest)

#let Colored(term, color: purple) = { emph(text(color)[#term]) }

#set math.equation(numbering: "(1)")
#set math.mat(delim: "[")
#set heading(numbering: "I.")
#show math.equation: set text(15pt)
#show heading: it => [
  #set align(left)
  #set text(font: "Times New Roman", blue)
  #counter(heading).display()
  #emph(it.body)
]

#import "@preview/showybox:2.0.1": showybox
// #show: amazed

#import "@preview/algo:0.3.3": algo, i, d, comment, code
// #import "@preview/cetz:0.2.0"
//
// #cetz.canvas({
//   import cetz.draw:*
// circle((0,0), anchor:"west")
// fill(red)
// stroke(none)
// circle((0,0),radius: 0.3)
//   })
// ---- Document

