import pandas as pd
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from text_to_speech import text_to_speech

read_expr = Expression.fromstring


class KnowledgeBaseInferencing:
    kb = []

    def __init__(self, kb_file='logical-kb.csv'):
        """
        Initializes the KnowledgeBaseInferencing class.

        Parameters:
        - kb_file: String, path to the file containing the logical knowledge base (default: 'logical-kb.csv').
        """
        self.read_expr = Expression.fromstring
        self.load_knowledge_base(kb_file)

    def load_knowledge_base(self, kb_file):
        """
        Loads the knowledge base from a CSV file.

        Parameters:
        - kb_file: String, path to the file containing the logical knowledge base.
        """
        kb_data = pd.read_csv(kb_file, header=None)

        for row in kb_data[0]:
            expr = self.read_expr(row)

            # Check for contradictions before adding to the knowledge base
            if not self.has_contradiction(expr):
                self.kb.append(expr)
            else:
                raise ValueError(f"Contradiction found: {row}. Terminating program.")

    def has_contradiction(self, new_expr):
        """
        Checks if adding a new expression to the knowledge base leads to a contradiction.

        Parameters:
        - new_expr: NLTK Expression, the new expression to be added to the knowledge base.

        Returns:
        - Boolean: True if a contradiction is found, False otherwise.
        """
        # Check if the negation of the new expression is already in the knowledge base
        negation_expr = self.read_expr('-' + str(new_expr))
        return negation_expr in self.kb

    def create_FOL_expressions(self, params, pattern):
        """
        Creates First-Order Logic expressions from user input.

        Parameters:
        - params: List, containing input parameters.
        - pattern: String, the pattern used to separate object and subject.

        Returns:
        - inputObject: String, object extracted from user input.
        - subject: String, subject extracted from user input.
        - expr: NLTK Expression, First-Order Logic expression created from input.
        """
        try:
            if pattern not in params[1]:
                raise ValueError(f"Invalid input format. Please use '{pattern}'")

            # Use the indexes to get the object and subject
            object = params[1].split(pattern)[0].lower()  # Words before the pattern (index 1)
            subject = params[1].split(pattern)[1].lower()  # Words after the pattern (index 2)

            # Check user input format
            if object.count(' ') > 0 or subject.count(' ') > 0:
                raise ValueError(f"Invalid input format: Please use one word before or after {pattern}")

            if pattern == ' IS ':
                expr = read_expr(subject + '(' + object + ')')
                return object, subject, expr

        except ValueError as e:
            print(e)
            return None, None, None

    def process_input(self, case, params, pattern, voiceEnabled):
        """
        Processes user input and performs inference based on the case.

        Parameters:
        - case: Integer, indicating the type of input case.
        - params: List, containing input parameters.
        - pattern: String, the pattern used to separate object and subject.
        - voiceEnabled: Boolean, indicating whether text-to-speech is enabled.
        """
        output = ''

        inputObject, subject, expr = self.create_FOL_expressions(params, pattern)

        if inputObject is None or subject is None or expr is None:
            return

        if case == 31:  # statements with "I know that"
            # Check for contradictions before appending to the knowledge base
            result_positive = ResolutionProver().prove(expr, self.kb, verbose=False)
            result_negative = ResolutionProver().prove(Expression.fromstring('-' + str(expr)), self.kb, verbose=False)

            if result_positive:
                output = "You are correct."
            elif result_negative:
                output = "That is incorrect."
            else:
                self.kb.append(expr)
                output = f"OK, I will remember that {inputObject} is {subject}"

        elif case == 32:  # statements with "check that"
            result_positive = ResolutionProver().prove(expr, self.kb, verbose=False)
            result_negative = ResolutionProver().prove(Expression.fromstring('-' + str(expr)), self.kb, verbose=False)

            if result_positive:
                output = 'The statement is correct based on my current knowledge base.'
            elif result_negative:
                output = 'The statement is incorrect based on my current knowledge base.'
            else:
                output = "Sorry, I don't know that."

        print(output)
        text_to_speech(voiceEnabled, output)
