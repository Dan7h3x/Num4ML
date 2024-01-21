#let title = [
  Project of Group 2
]
#set page(paper: "a4", margin: (x: 1.6cm, y: 1cm), numbering: "(1/1)")
#let Colored(term, color: purple) = { emph(text(color)[#term]) }
#import "config.typ": *

#align(center, text(22pt)[*#title*])

= Description
Suppose you have a databse of $n_(p)$ person photographed in $n_(e)$ different
expressions was represented by $p$ matrices $bold(A)_(p) in RR^(n_(i) times n_(e))$,
where $n_(i)$ is the number of pixels of each image. The same databse can be
considered as $cal(bold(A)) in RR^(n_(i) times n_(e) times n_(p))$.
#figure(image("FaceTensor.png", width: 50%), outlined: true)

= Face Recognition using HOSVD
Consider $bold(cal(A))$ is a databse. Using the #emph(text(font: "JetBrainsMono Nerd Font")[HOSVD]) method,
we have
$
  bold(cal(A)) = bold(S) times_(i) F times_(e) G times_(p) H,
$
where $times_(i),times_(e),times_(p)$ are the $1,2,3$-mode
production,respectively. For face recognition porpuse, let
$
  bold(cal(A)) = cal(C) times_(p) H, "where" cal(C) := cal(S) times_(i) F times_(e) G.
$

If we fix the expression $e$ and we identify the tensors $cal(A)(:,e,:)$ and $cal(C)(:,e,:)$ with
matrices $A_(e)$ and $C_(e)$ has relation as

$
  A_(e) = C_(e) H^(T), space space e=1,2,dots.c,n_(e).
$
Each column of $A_(e)$ contains the image of the person $p$ in the expression $e$.

Let $z in RR^(n_(i))$ be the image of an unknown person in an unknown expression
that we want to classify. The identification problem is

$
  limits("min")_(alpha_(e)) space norm(C_(e) alpha_(e) - z)_(2), space e=1,2,dots.c,n_(e),
$
obviously if $z$ is an image of a person $p$ in expression $e$, then the $alpha_(e)$ are
equal to $h_(p) = H(p,:)$.

The second approach is using the transposed version of $z$ as $hat(z) = F^(T) z$,
and the taking QR-decomposition of $cal(C)$, then solving $R alpha_(e) = Q^(T) hat(z)$ for
achieving faster.

Use extended Yale Database for apply the face recognition task using described
method. Please report the performance metrics of both methods on the dataset.
Compare and interpret the results. For the basics of tensors and #Colored([HOSVD], color: green.darken(52%)) or #Colored([HOOI], color: green.darken(50%)) please
read #underline("GeneralIntro.pdf").

