"""
dictionary.py - Bangla Trie Lexicon & Levenshtein Post-Processing Engine

Provides Trie-based vocabulary search, exact dictionary validation,
and Levenshtein distance dynamic programming fuzzy matching for Bangla offline HTR.
"""

import os
from typing import List, Set, Tuple, Optional, Dict


class TrieNode:
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False


class BanglaTrie:
    """Trie data structure for fast prefix checking and vocabulary lookup in Bangla."""

    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0

    def insert(self, word: str) -> None:
        """Inserts a word into the Trie."""
        word = word.strip()
        if not word:
            return
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1

    def search(self, word: str) -> bool:
        """Checks if a word exists in the dictionary."""
        word = word.strip()
        if not word:
            return False
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Checks if any word in the Trie starts with the given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


def levenshtein_distance(str1: str, str2: str) -> int:
    """Calculates Levenshtein edit distance between two strings using dynamic programming."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]    # Substitution
                )

    return dp[m][n]


class BanglaPostProcessor:
    """Post-processing engine combining Trie dictionary lookup and Levenshtein fuzzy matching."""

    def __init__(self, dictionary_path: Optional[str] = None, custom_words: Optional[List[str]] = None):
        self.trie = BanglaTrie()
        self.vocab_words: Set[str] = set()

        # Comprehensive baseline Bangla vocabulary
        default_words = [
            # Nations, Cities & Nature
            "বাংলা", "বাংলাদেশ", "ঢাকা", "চট্টগ্রাম", "সিলেট", "খুলনা", "রাজশাহী", "বরিশাল", "রংপুর", "ময়মনসিংহ",
            "ভাষা", "লেখা", "শব্দ", "বর্ণ", "শিক্ষা", "বিদ্যালয়", "মহাবিদ্যালয়", "বিশ্ববিদ্যালয়", "ছাত্র", "শিক্ষক",
            "বই", "খাতা", "কলম", "পেন্সিল", "কাগজ", "পত্রিকা", "জ্ঞান", "বিজ্ঞান", "প্রযুক্তি", "গণিত", "ইতিহাস",
            "সাহিত্য", "কবিতা", "গল্প", "উপন্যাস", "নাটক", "গান", "সুর", "শিল্পী", "ছবি", "রং", "আকাশ",
            "বাতাস", "পানি", "মাটি", "আগুন", "নদী", "পাহাড়", "সমুদ্র", "বন", "ফুল", "ফল", "গাছ", "পাখি", "পশু",
            # Family & Relationships
            "মানুষ", "বন্ধু", "পরিবার", "মা", "বাবা", "ভাই", "বোন", "সন্তান", "ছেলে", "মেয়ে", "স্বামী", "স্ত্রী",
            "পিতা", "মাতা", "দাদা", "দাদি", "নানা", "নানি", "চাচা", "খালা", "মামা", "ফুফু", "আত্মীয়",
            # Home, Society & Places
            "ঘর", "বাড়ি", "গ্রাম", "শহর", "দেশ", "পৃথিবী", "সূর্য", "চাঁদ", "তারা", "বায়ু", "বৃষ্টি", "মেঘ",
            "দিন", "রাত", "সকাল", "দুপুর", "সন্ধ্যা", "সময়", "বছর", "মাস", "সপ্তাহ", "আজ", "গতকাল", "আগামীকাল",
            "ঘণ্টা", "মিনিট", "সেকেন্ড", "ঋতু", "গ্রীষ্ম", "বর্ষা", "শরৎ", "হেমেন্ত", "শীত", "বসন্ত",
            # Activities & Concepts
            "কাজ", "জীবন", "আশা", "স্বপ্ন", "আনন্দ", "ভালোবাসা", "শান্তি", "সত্য", "সুন্দর", "নতুন", "পুরাতন",
            "ভালো", "মন্দ", "বড়", "ছোট", "উচ্চ", "নিম্ন", "সহজ", "কঠিন", "প্রথম", "শেষ", "স্বাধীনতা", "সংগ্রাম",
            "সংস্কৃতি", "ঐতিহ্য", "উৎসব", "মেলা", "খেলারাম", "খেলাধুলা", "ফুটবল", "ক্রিকেট", "বিজয়", "পরাজয়",
            # Numbers
            "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ", "শত", "হাজার", "লক্ষ", "কোটি",
            "প্রথম", "দ্বিতীয়", "তৃতীয়", "চতুর্থ", "পঞ্চম", "ষষ্ঠ", "সপ্তম", "অষ্টম", "নবম", "দশম",
            # Common Verbs & Adjectives
            "করা", "হওয়া", "যাওয়া", "আসা", "বলা", "শোনা", "দেখা", "পড়া", "লেখা", "খাওয়া", "শোয়া", "বসা",
            "হাঁটা", "চলা", "দেওয়া", "নেওয়া", "ভাবা", "বোঝা", "শেখা", "জানা", "মানামানি", "চিন্তা", "চেষ্টা",
            "মহান", "পবিত্র", "উজ্জ্বল", "গভীর", "শান্ত", "ধীর", "দ্রুত", "সহজ", "কঠিন", "মিষ্টি", "তিতা", "ঝাল"
        ]

        for word in default_words:
            self.add_word(word)

        if custom_words:
            for word in custom_words:
                self.add_word(word)

        # Check for local dictionary text file if provided or if bangla_dictionary.txt exists
        dict_file = dictionary_path or "bangla_dictionary.txt"
        if os.path.isfile(dict_file):
            self.load_dictionary_file(dict_file)

    def add_word(self, word: str) -> None:
        word = word.strip()
        if word:
            self.trie.insert(word)
            self.vocab_words.add(word)

    def load_dictionary_file(self, file_path: str) -> int:
        """Loads words from a plain text file (one word per line). Returns total words added."""
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    self.add_word(word)
                    count += 1
        return count

    def is_valid_word(self, word: str) -> bool:
        """Returns True if the predicted word is in the dictionary."""
        return self.trie.search(word)

    def correct_word(self, predicted_word: str, max_edit_distance: int = 2) -> Tuple[str, int]:
        """Corrects an out-of-vocabulary word using minimum Levenshtein distance."""
        predicted_word = predicted_word.strip()
        if not predicted_word or self.is_valid_word(predicted_word):
            return predicted_word, 0

        best_candidate = predicted_word
        min_dist = float('inf')

        # Fast match filter by length similarity
        p_len = len(predicted_word)
        for word in self.vocab_words:
            if abs(len(word) - p_len) > max_edit_distance:
                continue
            dist = levenshtein_distance(predicted_word, word)
            if dist < min_dist:
                min_dist = dist
                best_candidate = word
                if dist == 1:
                    break

        if min_dist <= max_edit_distance:
            return best_candidate, min_dist
        return predicted_word, 0

    def correct_sentence(self, sentence: str) -> str:
        """Processes a sentence word by word and applies dictionary correction."""
        words = sentence.strip().split()
        corrected_words = []
        for word in words:
            corr_w, _ = self.correct_word(word)
            corrected_words.append(corr_w)
        return " ".join(corrected_words)


if __name__ == "__main__":
    post_proc = BanglaPostProcessor()
    print(f"Total built-in dictionary vocabulary: {len(post_proc.vocab_words)} words")
    test_word = "বাংলাদশে"
    is_valid = post_proc.is_valid_word(test_word)
    corrected, dist = post_proc.correct_word(test_word)
    print(f"Original: '{test_word}' | Valid: {is_valid} | Corrected: '{corrected}' (Distance: {dist})")
