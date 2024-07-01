from abc import ABC, abstractmethod

from compiler.program import CommonPreprocessedInput
from curve import G1Point
from poly import Polynomial
from verifier import VerificationKey


class CommitmentScheme(ABC):

    @abstractmethod
    def commit(self, values: Polynomial) -> G1Point:
        raise NotImplementedError

    @abstractmethod
    def verification_key(self, pk: CommonPreprocessedInput) -> VerificationKey:
        raise NotImplementedError
