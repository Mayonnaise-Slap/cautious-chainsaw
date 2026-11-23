class INT4LinearWeights:

    def __init__(self, P, S, N, K, K_PAD, group_size=128):

        self.P = P
        self.S = S

        self.N = N
        self.K = K
        self.K_PAD = K_PAD
        self.group_size = group_size

        self.W_fp16 = None

    def prepare(self):

        if self.W_fp16 is None:

            self.W_fp16 = dequant_int4(
                self.P,
                self.S,
                self.N,
                self.K,
                self.K_PAD,
                self.group_size
            )

        return self.W_fp16

    def forward(self, X):

        W = self.prepare()

        return X @ W.T