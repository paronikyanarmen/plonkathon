from commitment_scheme import CommitmentScheme
from compiler.program import CommonPreprocessedInput
from curve import G1Point, ec_lincomb, Scalar
from poly import Polynomial, Basis
from setup import Setup
from verifier import VerificationKey


class KateCommitment(CommitmentScheme):
    setup: Setup

    def __init__(self, setup: Setup):
        self.setup = setup

    def commit(self, values: Polynomial) -> G1Point:
        assert values.basis == Basis.LAGRANGE

        coeffs = values.ifft().values

        if len(coeffs) > len(self.setup.powers_of_x):
            raise ValueError()

        commitment = ec_lincomb(list(zip(self.setup.powers_of_x, coeffs)))

        # Run inverse FFT to convert values from Lagrange basis to monomial basis
        # Optional: Check values size does not exceed maximum power setup can handle
        # Compute linear combination of setup with values
        return commitment

    def verification_key(self, pk: CommonPreprocessedInput) -> VerificationKey:
        vk = VerificationKey(
            group_order=pk.group_order,
            Ql=self.commit(pk.QL),
            Qr=self.commit(pk.QR),
            Qo=self.commit(pk.QO),
            Qm=self.commit(pk.QM),
            Qc=self.commit(pk.QC),
            S1=self.commit(pk.S1),
            S2=self.commit(pk.S2),
            S3=self.commit(pk.S3),
            w=Scalar.root_of_unity(pk.group_order),
            X_2=self.setup.X2
        )
        # Create the appropriate VerificationKey object
        return vk
