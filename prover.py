import random

from commitment_scheme import CommitmentScheme
from compiler.program import Program, CommonPreprocessedInput
from utils import *
from setup import *
from typing import Optional
from dataclasses import dataclass
from transcript import Transcript, Message1, Message2, Message3, Message4, Message5
from poly import Polynomial, Basis


@dataclass
class Proof:
    msg_1: Message1
    msg_2: Message2
    msg_3: Message3
    msg_4: Message4
    msg_5: Message5

    def flatten(self):
        proof = {}
        proof["a_1"] = self.msg_1.a_1
        proof["b_1"] = self.msg_1.b_1
        proof["c_1"] = self.msg_1.c_1
        proof["z_1"] = self.msg_2.z_1
        proof["t_lo_1"] = self.msg_3.t_lo_1
        proof["t_mid_1"] = self.msg_3.t_mid_1
        proof["t_hi_1"] = self.msg_3.t_hi_1
        proof["a_eval"] = self.msg_4.a_eval
        proof["b_eval"] = self.msg_4.b_eval
        proof["c_eval"] = self.msg_4.c_eval
        proof["s1_eval"] = self.msg_4.s1_eval
        proof["s2_eval"] = self.msg_4.s2_eval
        proof["z_shifted_eval"] = self.msg_4.z_shifted_eval
        proof["W_z_1"] = self.msg_5.W_z_1
        proof["W_zw_1"] = self.msg_5.W_zw_1
        return proof


@dataclass
class Prover:
    group_order: int
    program: Program
    pk: CommonPreprocessedInput
    commitment_scheme: CommitmentScheme

    def __init__(self, commitment_scheme: CommitmentScheme, program: Program):
        self.group_order = program.group_order
        self.commitment_scheme = commitment_scheme
        self.program = program
        self.pk = program.common_preprocessed_input()

    def prove(self, witness: dict[Optional[str], int]) -> Proof:
        # Initialise Fiat-Shamir transcript
        transcript = Transcript(b"plonk")

        # Collect fixed and public information
        # FIXME: Hash pk and PI into transcript
        public_vars = self.program.get_public_assignments()
        PI = Polynomial(
            [Scalar(-witness[v]) for v in public_vars]
            + [Scalar(0) for _ in range(self.group_order - len(public_vars))],
            Basis.LAGRANGE,
        )
        self.PI = PI

        # Round 1
        msg_1 = self.round_1(witness)
        self.beta, self.gamma = transcript.round_1(msg_1)

        # Round 2
        msg_2 = self.round_2()
        self.alpha, self.fft_cofactor = transcript.round_2(msg_2)

        # Round 3
        msg_3 = self.round_3()
        self.zeta = transcript.round_3(msg_3)

        # Round 4
        msg_4 = self.round_4()
        self.v = transcript.round_4(msg_4)

        # Round 5
        msg_5 = self.round_5()

        return Proof(msg_1, msg_2, msg_3, msg_4, msg_5)

    def round_1(
        self,
        witness: dict[Optional[str], int],
    ) -> Message1:
        program = self.program
        group_order = self.group_order

        if None not in witness:
            witness[None] = 0

        # Compute wire assignments for A, B, C, corresponding:
        # - A_values: witness[program.wires()[i].L]
        # - B_values: witness[program.wires()[i].R]
        # - C_values: witness[program.wires()[i].O]
        A_values = [Scalar(0)] * group_order
        B_values = [Scalar(0)] * group_order
        C_values = [Scalar(0)] * group_order
        for i, wire in enumerate(program.wires()):
            A_values[i] = Scalar(witness[wire.L])
            B_values[i] = Scalar(witness[wire.R])
            C_values[i] = Scalar(witness[wire.O])

        # Construct A, B, C Lagrange interpolation polynomials for
        # A_values, B_values, C_values
        roots_of_unity = Scalar.roots_of_unity(group_order)

        Z_H = Polynomial(
            [r**group_order - 1 for r in roots_of_unity],
            Basis.LAGRANGE
        )

        roots = Polynomial(roots_of_unity, basis=Basis.LAGRANGE)

        blinding_terms = [0] * 6
        for i in range(6):
            blinding_terms[i] = Scalar(random.randint(0, Scalar.field_modulus - 1))

        self.A = Polynomial(A_values, Basis.LAGRANGE) + Z_H * (roots * blinding_terms[0] + blinding_terms[1])
        self.B = Polynomial(B_values, Basis.LAGRANGE) + Z_H * (roots * blinding_terms[2] + blinding_terms[3])
        self.C = Polynomial(C_values, Basis.LAGRANGE) + Z_H * (roots * blinding_terms[4] + blinding_terms[5])

        # Compute a_1, b_1, c_1 commitments to A, B, C polynomials
        a_1 = self.commitment_scheme.commit(self.A)
        b_1 = self.commitment_scheme.commit(self.B)
        c_1 = self.commitment_scheme.commit(self.C)

        # Sanity check that witness fulfils gate constraints
        assert (
            self.A * self.pk.QL
            + self.B * self.pk.QR
            + self.A * self.B * self.pk.QM
            + self.C * self.pk.QO
            + self.PI
            + self.pk.QC
            == Polynomial([Scalar(0)] * group_order, Basis.LAGRANGE)
        )

        # Return a_1, b_1, c_1
        return Message1(a_1, b_1, c_1)

    def round_2(self) -> Message2:
        group_order = self.group_order

        # Using A, B, C, values, and pk.S1, pk.S2, pk.S3, compute
        # Z_values for permutation grand product polynomial Z
        #
        # Note the convenience function:
        #       self.rlc(val1, val2) = val_1 + self.beta * val_2 + gamma

        roots_of_unity = Scalar.roots_of_unity(group_order)

        Z_values = [Scalar(1)]
        k_1 = 2
        k_2 = 3

        for i in range(group_order):
            z_numerator = (
                self.rlc(self.A.values[i], roots_of_unity[i]) *
                self.rlc(self.B.values[i], k_1 * roots_of_unity[i]) *
                self.rlc(self.C.values[i], k_2 * roots_of_unity[i])
            )

            z_denominator = (
                self.rlc(self.A.values[i], self.pk.S1.values[i]) *
                self.rlc(self.B.values[i], self.pk.S2.values[i]) *
                self.rlc(self.C.values[i], self.pk.S3.values[i])
            )

            Z_values.append(Z_values[-1] * z_numerator / z_denominator)

        # Check that the last term Z_n = 1
        assert Z_values.pop() == 1

        # Sanity-check that Z was computed correctly
        for i in range(group_order):
            assert (
                self.rlc(self.A.values[i], roots_of_unity[i])
                * self.rlc(self.B.values[i], 2 * roots_of_unity[i])
                * self.rlc(self.C.values[i], 3 * roots_of_unity[i])
            ) * Z_values[i] - (
                self.rlc(self.A.values[i], self.pk.S1.values[i])
                * self.rlc(self.B.values[i], self.pk.S2.values[i])
                * self.rlc(self.C.values[i], self.pk.S3.values[i])
            ) * Z_values[
                (i + 1) % group_order
            ] == 0

        roots_of_unity = Scalar.roots_of_unity(group_order)

        Z_H = Polynomial(
            [r**group_order - 1 for r in roots_of_unity],
            Basis.LAGRANGE
        )

        roots = Polynomial(roots_of_unity, basis=Basis.LAGRANGE)
        roots_squared = Polynomial(
            [r**2 for r in roots_of_unity],
            Basis.LAGRANGE
        )

        blinding_terms = [0] * 3
        for i in range(3):
            blinding_terms[i] = Scalar(random.randint(0, Scalar.field_modulus - 1))

        # Construct Z, Lagrange interpolation polynomial for Z_values
        blinding_part = Z_H * (roots_squared * blinding_terms[0] + roots * blinding_terms[1] + blinding_terms[2])

        self.Z = Polynomial(Z_values, Basis.LAGRANGE) + blinding_part

        # Cpmpute z_1 commitment to Z polynomial
        z_1 = self.commitment_scheme.commit(self.Z)

        # Return z_1
        return Message2(z_1)

    def round_3(self) -> Message3:
        group_order = self.group_order

        # Compute the quotient polynomial

        # List of roots of unity at 4x fineness, i.e. the powers of µ
        # where µ^(4n) = 1

        # Using self.fft_expand, move A, B, C into coset extended Lagrange basis
        A_big = self.fft_expand(self.A)
        B_big = self.fft_expand(self.B)
        C_big = self.fft_expand(self.C)

        # Expand public inputs polynomial PI into coset extended Lagrange
        PI_big = self.fft_expand(self.PI)

        # Expand selector polynomials pk.QL, pk.QR, pk.QM, pk.QO, pk.QC
        # into the coset extended Lagrange basis
        QL = self.fft_expand(self.pk.QL)
        QR = self.fft_expand(self.pk.QR)
        QO = self.fft_expand(self.pk.QO)
        QM = self.fft_expand(self.pk.QM)
        QC = self.fft_expand(self.pk.QC)

        # Expand permutation grand product polynomial Z into coset extended
        # Lagrange basis

        Z_big = self.fft_expand(self.Z)

        # Expand shifted Z(ω) into coset extended Lagrange basis
        Z_shifted_big = Z_big.shift(4)

        # Expand permutation polynomials pk.S1, pk.S2, pk.S3 into coset
        # extended Lagrange basis
        S1 = self.fft_expand(self.pk.S1)
        S2 = self.fft_expand(self.pk.S2)
        S3 = self.fft_expand(self.pk.S3)

        # Compute Z_H = X^N - 1, also in evaluation form in the coset

        # Compute L0, the Lagrange basis polynomial that evaluates to 1 at x = 1 = ω^0
        # and 0 at other roots of unity
        self.L0 = Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), basis=Basis.LAGRANGE)

        # Expand L0 into the coset extended Lagrange basis
        L0_big = self.fft_expand(self.L0)

        # Compute the quotient polynomial (called T(x) in the paper)
        # It is only possible to construct this polynomial if the following
        # equations are true at all roots of unity {1, w ... w^(n-1)}:
        # 1. All gates are correct:
        #    A * QL + B * QR + A * B * QM + C * QO + PI + QC = 0
        #
        # 2. The permutation accumulator is valid:
        #    Z(wx) = Z(x) * (rlc of A, X, 1) * (rlc of B, 2X, 1) *
        #                   (rlc of C, 3X, 1) / (rlc of A, S1, 1) /
        #                   (rlc of B, S2, 1) / (rlc of C, S3, 1)
        #    rlc = random linear combination: term_1 + beta * term2 + gamma * term3
        #
        # 3. The permutation accumulator equals 1 at the start point
        #    (Z - 1) * L0 = 0
        #    L0 = Lagrange polynomial, equal at all roots of unity except 1

        gate_constraints = (A_big * B_big * QM + A_big * QL + B_big * QR + C_big * QO + PI_big + QC)

        quarter_roots = Scalar.roots_of_unity(4 * group_order)

        Z_H_big = Polynomial(
            [
                ((Scalar(r) * self.fft_cofactor) ** group_order - 1)
                for r in quarter_roots
            ],
            Basis.LAGRANGE,
        )
        quarter_roots = Polynomial(quarter_roots, basis=Basis.LAGRANGE)
        k_1 = 2
        k_2 = 3
        permutation_accumulator = (
            (
                self.rlc(A_big, quarter_roots * self.fft_cofactor) *
                self.rlc(B_big, quarter_roots * (self.fft_cofactor * k_1)) *
                self.rlc(C_big, quarter_roots * (self.fft_cofactor * k_2)) * Z_big
            ) -
            (
                self.rlc(A_big, S1) *
                self.rlc(B_big, S2) *
                self.rlc(C_big, S3) * Z_shifted_big
            )
        )

        permutation_final_check = (Z_big - Scalar(1)) * L0_big

        QUOT_big = (
            gate_constraints +
            permutation_accumulator * self.alpha +
            permutation_final_check * self.alpha ** 2
        ) / Z_H_big

        QUOT_coeffs = self.expanded_evals_to_coeffs(QUOT_big)

        # Sanity check: QUOT has degree < 3n
        assert (
            QUOT_coeffs.values[-group_order:]
            == [0] * group_order
        )
        print("Generated the quotient polynomial")

        # Split up T into T1, T2 and T3 (needed because T has degree 3n - 4, so is
        # too big for the trusted setup)
        self.T1 = Polynomial(QUOT_coeffs.values[:group_order], basis=Basis.MONOMIAL).fft()
        self.T2 = Polynomial(QUOT_coeffs.values[group_order: 2 * group_order], basis=Basis.MONOMIAL).fft()
        self.T3 = Polynomial(QUOT_coeffs.values[2 * group_order: 3 * group_order], basis=Basis.MONOMIAL).fft()

        # Sanity check that we've computed T1, T2, T3 correctly
        assert (
            self.T1.barycentric_eval(self.fft_cofactor)
            + self.T2.barycentric_eval(self.fft_cofactor) * self.fft_cofactor**group_order
            + self.T3.barycentric_eval(self.fft_cofactor) * self.fft_cofactor ** (group_order * 2)
        ) == QUOT_big.values[0]

        print("Generated T1, T2, T3 polynomials")

        # Compute commitments t_lo_1, t_mid_1, t_hi_1 to T1, T2, T3 polynomials
        t_lo_1 = self.commitment_scheme.commit(self.T1)
        t_mid_1 = self.commitment_scheme.commit(self.T2)
        t_hi_1 = self.commitment_scheme.commit(self.T3)

        # Return t_lo_1, t_mid_1, t_hi_1
        return Message3(t_lo_1, t_mid_1, t_hi_1)

    def round_4(self) -> Message4:
        # Compute evaluations to be used in constructing the linearization polynomial.

        # Compute a_eval = A(zeta)
        self.a_eval = self.A.barycentric_eval(self.zeta)
        # Compute b_eval = B(zeta)
        self.b_eval = self.B.barycentric_eval(self.zeta)
        # Compute c_eval = C(zeta)
        self.c_eval = self.C.barycentric_eval(self.zeta)
        # Compute s1_eval = pk.S1(zeta)
        self.s1_eval = self.pk.S1.barycentric_eval(self.zeta)
        # Compute s2_eval = pk.S2(zeta)
        self.s2_eval = self.pk.S2.barycentric_eval(self.zeta)
        # Compute z_shifted_eval = Z(zeta * ω)
        self.z_shifted_eval = self.Z.barycentric_eval(self.zeta * Scalar.root_of_unity(group_order=self.group_order))
        # Return self.a_eval, b_eval, c_eval, s1_eval, s2_eval, z_shifted_eval
        return Message4(self.a_eval, self.b_eval, self.c_eval, self.s1_eval, self.s2_eval, self.z_shifted_eval)

    def round_5(self) -> Message5:
        group_order = self.group_order

        # Evaluate the Lagrange basis polynomial L0 at zeta
        L0_eval = self.L0.barycentric_eval(self.zeta)

        # Evaluate the vanishing polynomial Z_H(X) = X^n - 1 at zeta
        Z_H_eval = self.zeta**self.group_order - 1

        PI_eval = self.PI.barycentric_eval(self.zeta)

        # Move T1, T2, T3 into the coset extended Lagrange basis
        T1_big = self.fft_expand(self.T1)
        T2_big = self.fft_expand(self.T2)
        T3_big = self.fft_expand(self.T3)

        # Move pk.QL, pk.QR, pk.QM, pk.QO, pk.QC into the coset extended Lagrange basis
        QL = self.fft_expand(self.pk.QL)
        QR = self.fft_expand(self.pk.QR)
        QO = self.fft_expand(self.pk.QO)
        QM = self.fft_expand(self.pk.QM)
        QC = self.fft_expand(self.pk.QC)

        # Move Z into the coset extended Lagrange basis
        Z_big = self.fft_expand(self.Z)
        # Move pk.S3 into the coset extended Lagrange basis
        S3_big = self.fft_expand(self.pk.S3)

        # Compute the "linearization polynomial" R. This is a clever way to avoid
        # needing to provide evaluations of _all_ the polynomials that we are
        # checking an equation between: instead, we can "skip" the first
        # multiplicand in each term. The idea is that we construct a
        # polynomial which is constructed to equal 0 at Z only if the equations
        # that we are checking are correct, and which the verifier can reconstruct
        # the KZG commitment to, and we provide proofs to verify that it actually
        # equals 0 at Z
        #
        # In order for the verifier to be able to reconstruct the commitment to R,
        # it has to be "linear" in the proof items, hence why we can only use each
        # proof item once; any further multiplicands in each term need to be
        # replaced with their evaluations at Z, which do still need to be provided

        gate_constraints = (
            QM * self.a_eval * self.b_eval +
            QL * self.a_eval +
            QR * self.b_eval +
            QO * self.c_eval +
            QC +
            PI_eval
        )

        c_eval = Polynomial([self.c_eval] * self.group_order * 4, Basis.LAGRANGE)

        permutation_polynomial = (
            Z_big * (
                self.rlc(self.a_eval, self.zeta) *
                self.rlc(self.b_eval, 2 * self.zeta) *
                self.rlc(self.c_eval, 3 * self.zeta)
            ) -
            (
                self.rlc(c_eval, S3_big) *
                self.rlc(self.a_eval, self.s1_eval) *
                self.rlc(self.b_eval, self.s2_eval)
            ) * self.z_shifted_eval
        ) * self.alpha

        permutation_final_check = (Z_big - Scalar(1)) * L0_eval * self.alpha**2

        final_term = (
            T1_big +
            T2_big * self.zeta**self.group_order +
            T3_big * self.zeta**(2*self.group_order)
        ) * Z_H_eval

        R_big = gate_constraints + permutation_polynomial + permutation_final_check - final_term

        R_coeffs = self.expanded_evals_to_coeffs(R_big).values
        assert R_coeffs[self.group_order:] == [0] * (self.group_order * 3)

        R = Polynomial(R_coeffs[:self.group_order], basis=Basis.MONOMIAL).fft()


        # Sanity-check R
        assert R.barycentric_eval(self.zeta) == 0

        print("Generated linearization polynomial R")

        # Generate proof that W(z) = 0 and that the provided evaluations of
        # A, B, C, S1, S2 are correct

        # Move A, B, C into the coset extended Lagrange basis
        # Move pk.S1, pk.S2 into the coset extended Lagrange basis

        A_big = self.fft_expand(self.A)
        B_big = self.fft_expand(self.B)
        C_big = self.fft_expand(self.C)

        S1 = self.fft_expand(self.pk.S1)
        S2 = self.fft_expand(self.pk.S2)

        # In the COSET EXTENDED LAGRANGE BASIS,
        # Construct W_Z = (
        #     R
        #   + v * (A - a_eval)
        #   + v**2 * (B - b_eval)
        #   + v**3 * (C - c_eval)
        #   + v**4 * (S1 - s1_eval)
        #   + v**5 * (S2 - s2_eval)
        # ) / (X - zeta)
        quarter_roots = Scalar.roots_of_unity(4 * group_order)
        X_minus_zeta = Polynomial([r * self.fft_cofactor - self.zeta for r in quarter_roots], Basis.LAGRANGE)

        W_Z_big = (
            R_big +
            (A_big - self.a_eval) * self.v +
            (B_big - self.b_eval) * self.v**2 +
            (C_big - self.c_eval) * self.v**3 +
            (S1 - self.s1_eval) * self.v**4 +
            (S2 - self.s2_eval) * self.v**5
        ) / X_minus_zeta

        W_z_coeffs = self.expanded_evals_to_coeffs(W_Z_big).values

        # Check that degree of W_z is not greater than n
        assert W_z_coeffs[group_order:] == [0] * (group_order * 3)

        W_z = Polynomial(W_z_coeffs[:group_order], Basis.MONOMIAL).fft()

        # Compute W_z_1 commitment to W_z
        W_z_1 = self.commitment_scheme.commit(W_z)

        # Generate proof that the provided evaluation of Z(z*w) is correct. This
        # awkwardly different term is needed because the permutation accumulator
        # polynomial Z is the one place where we have to check between adjacent
        # coordinates, and not just within one coordinate.
        # In other words: Compute W_zw = (Z - z_shifted_eval) / (X - zeta * ω)

        omega = Scalar.root_of_unity(self.group_order)

        X_minus_zeta_w = Polynomial([r * self.fft_cofactor - self.zeta * omega for r in quarter_roots], Basis.LAGRANGE)

        W_zw_big = (Z_big - self.z_shifted_eval) / X_minus_zeta_w

        W_zw_coeffs = self.expanded_evals_to_coeffs(W_zw_big).values

        # Check that degree of W_z is not greater than n
        assert W_zw_coeffs[group_order:] == [0] * (group_order * 3)

        # Compute W_z_1 commitment to W_z

        W_zw = Polynomial(W_zw_coeffs[:group_order], Basis.MONOMIAL).fft()

        W_zw_1 = self.commitment_scheme.commit(W_zw)

        print("Generated final quotient witness polynomials")

        # Return W_z_1, W_zw_1
        return Message5(W_z_1, W_zw_1)

    def fft_expand(self, x: Polynomial):
        return x.to_coset_extended_lagrange(self.fft_cofactor)

    def expanded_evals_to_coeffs(self, x: Polynomial):
        return x.coset_extended_lagrange_to_coeffs(self.fft_cofactor)

    def rlc(self, term_1, term_2):
        return term_1 + term_2 * self.beta + self.gamma
