#let title = [
  General Introduction of Tensors
]
#set page(
  paper: "a4", margin: (x: 1.6cm, y: 1cm), header: align(right + horizon)[
    *#title*
  ], numbering: "(1/1)",
)

#import "config.typ": *

// ---- Document

#align(center, text(22pt)[*#title*])

= Introduction

#showybox(
  frame: (border-color: red, body-color: lime.lighten(65%)), [In *Linear Algebra*, $bold(a) in RR^I$ is a vector, $bold(A) in RR^(I times J)$ is
    a matrix, and $cal(bold(A)) in RR^(I_1 times I_2 times dots.c times I_N)$ is a $N$-mode
    tensor. The $i$th, ${i,j}$th and ${i_1,...,i_N}$th entries of $cal(bold(A))$ are
    denoted by $bold(a)(i)$, $bold(A)(i,j)$ and $cal(bold(A))(i_1,...,i_N)$.],
)

== Products
For matrices $bold(A) "and" bold(B)$ with same size $I times J$, the $bold("Hadamard")$ product
is the element-wise product as:

$ bold(C) = bold(A) ast.circle bold(B) in RR^(I times J)
= mat(
  a_(1,1) b_(1,1), a_(1,2) b_(1,2), dots.c, a_(1,J) b_(1,J);a_(2,1) b_(2,1), a_(2,2) b_(2,2), dots.c, a_(2,1) b_(2,J);dots.v, dots.v, dots.down, dots.v;a_(I,1) b_(I,1), a_(I,2) b_(I,2), dots.c, a_(I,J) b_(I,J);
), $

and the $bold("Kronecker")$ product with $bold(D) in RR^(K times L)$ is:

$
  bold(E) = bold(A) times.circle bold(D) in RR^(I K times J L)
  = mat(
    a_(1,1) bold(D), a_(1,2) bold(D), dots.c, a_(1,J) bold(D);a_(2,1) bold(D), a_(2,2) bold(D), dots.c, a_(2,1) bold(D);dots.v, dots.v, dots.down, dots.v;a_(I,1) bold(D), a_(I,2) bold(D), dots.c, a_(I,J) bold(D);
  ),
$
and $bold("Khatri-Rao")$ product of $bold(A) in RR^(I times K) "and" bold(B) in RR^(J times K)$ is
defined as:

$
  bold(A) dot.circle bold(B) = [bold(a)_1 times.circle bold(b)_1,bold(a)_2 times.circle bold(b)_2, dots.c,bold(a)_K times.circle bold(b)_K] in RR^(I J times K),
$
where $bold(a)_k "and" bold(b)_k$ are the $k$th column of matrices $bold(A) "and" bold(B)$.

The $bold("Mode-n")$ product of tensor $cal(bold(A)) in RR^(I_(1) times dots.c times I_N)$ and
matrix $bold(B) in RR^(J times I_n)$ is defined by

$
  bold(cal(C)) = bold(cal(A)) times_(n) bold(B) in RR^(I_(1) times dots.c times I_(n-1) times J times I_(n+1) times dots.c times I_N),
$
whose entries are
$
  bold(cal(C))(i_(1),dots.c, j ,dots.c , i_N) = sum_(i_n = 1)^(I_n) bold(cal(A))(i_(1),dots.c, i_n ,dots.c , i_N) bold(B)(j,i_n).
$

= Decompostions
$bold("Canonical Polyadic (CP)")$: For a $N$-mode tensor $cal(X) in RR^(I_(1) times dots.c times I_(N))$,
the CP decompostion is defined by
$
  cal(X) = sum_(r=1)^(R) bold(u)_(r)^(( 1 )) circle.small bold(u)_(r)^(( 2 )) circle.small space dots.c space circle.small bold(u)_(r)^(( N )) = bracket.double.l bold(U)^((1)),bold(U)^((2)), dots.c ,bold(U)^((N)) bracket.double.r,
$
where $bold(U)^((n)) = [bold(u)_(r)^(( 1 )),bold(u)_(r)^(( 2 )),dots,bold(u)_(R)^(( n ))] in RR^(I_(n) times R), n = 1,dots,N$ are
factor matrices.

$bold("Tucker")$:
$
  cal(X) = cal(G) times_1 bold(U)^((1)) times_2 bold(U)^((2)) dots.c times_N bold(U)^((N)) = bracket.double.l cal(G);bold(U)^((1)),bold(U)^((2)), dots.c ,bold(U)^((N)) bracket.double.r,
$
where $cal(G) in RR^(R_(1) times R_(2) times dots.c times R_(N))$ is called #emph(text(red, style: "italic")[core]) tensor.

== The $bold("n-Rank")$
#showybox(
  frame: (border-color: orange, body-color: gray.lighten(75%)), [
    For $cal(X) in RR^(I_(1) times dots.c times I_(N))$, the $n$-rank of $cal(X)$,
    denoted by $"rank"_(n)(cal(X))$ is the column rank of $cal(X)_((n))$ which is
    the size of the vector space spanned by the mode-$n$ fibers. If $R_(n) = "rank"_(n)(cal(X)), space n=1,dots,N$,
    we can say that $cal(X)$ is a rank-$(R_(1),R_(2),dots,R_(N))$ tensor
  ],
)

= $bold("HOSVD")$

The method #strong("HOSVD") is convincing generalization of the matrix SVD and
capable of computing the left singular vectors of $cal(X)_((n))$. When $R_(n) < "rank"_(n)(cal(X))$ for
one or more $n$, the decompostion is called truncated HOSVD.

#algo(
  title: "HOSVD", radius: 10pt,
)[
  $bold("Input:") cal(X) in RR^(I_(1) times I_(2) times dots.c I_(N))$ and $n$-rank $(R_(1),R_(2),dots.c,R_(N))$ \
  $bold("Output:") cal(G),bold(U^((1))),bold(U^((2))),dots.c,bold(U^((N)))$\
  for $n=1,dots.c,N$ do #i\
  $bold(U^((n))) arrow.l R_(n) "Leading left singular vectors of" cal(X)_((n))$ #d\ #comment[End For]\

  return $cal(G) arrow.l cal(X) times_1 bold(U)^((1)T) times_2 bold(U)^((2)T)
  dots.c times_N bold(U)^((N)T)$
]

= $bold("HOOI")$
A good refinement of HOSVD method is the following constrained optimiziation
problem:

$
  limits("min")_(cal(G),bold(U^((1))),bold(U^((2))),dots.c,bold(U^((N)))) norm(
    ( cal(X) - bracket.double.l cal(G); bold(U^((1))), bold(U)^((2)), dots.c, bold(U^((N))) bracket.double.r )
  )_(F)^2 \
  "s.t." space cal(G) in RR^(R_(1),R_(2),dots.c,R_(N)), space bold(U^((n))) in RR^(I_(n) times R_(n)), space bold(U^((n)T)) bold(U^((n))) = bold(I)_(R_(n) times R_(n)).
$

where after some mathematical operations, the problem boils down to

$
  limits("max")_(bold(U^((n)))) norm(cal(X) times_(0) bold(U)^((1)T) dots.c times_(N) bold(U)^((N)T))_(F)^(2)\
  "s.t." space bold(U)^((n)) in RR^(I_(n) times R_(n)), space bold(U)^((n)T) bold(U)^((n)) = bold(I)_(R_(n) times R_(n)),
$

This formulation called *HOOI* method which has a protocol described in

#algo(
  title: "HOOI", radius: 10pt,
)[
  $bold("Input:") cal(X) in RR^(I_(1) times I_(2) times dots.c I_(N))$ and $n$-rank $(R_(1),R_(2),dots.c,R_(N))$ \
  $bold("Output:") cal(G),bold(U^((1))),bold(U^((2))),dots.c,bold(U^((N)))$\
  $bold("Initialize" U^((n))) in RR^(I_(n) times R_(n)) $ for $n=1,dots,N$ using
  HOSVD, \
  repeat\
  for $n=1,dots.c,N$ do #i\
  #v(10pt)
  $cal(Y) arrow.l cal(X) times_(1) bold(U)^((1)T) dots.c times_(n-1) bold(U)^((n-1)T) times_(n+1) bold(U)^((n+1)T) dots.c times_(N) bold(U)^((N)T)\
  bold(U^((n)))
  arrow.l R_(n) "Leading left singular vectors of" cal(Y)_((n))$ #d\ #comment[End for]\

  until fit ceases to improve or maximum iterations exhausted.\
  return $cal(G) arrow.l cal(X) times_1 bold(U)^((1)T) times_2 bold(U)^((2)T)
  dots.c times_N bold(U)^((N)T)$
]

#bibliography("projects.bib")
