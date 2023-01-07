from itertools import combinations
from tqdm import trange
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MenuSolver:
    def __init__(
        self,
        menu_df: pd.DataFrame,
        difficulty_dict: dict,
        n_days: int,
        n_candidates: int,
        n_fittest: int,
        n_iter: int,
        p_mutation: float,
    ):
        self.menu_df = menu_df
        self.difficulty_dict = difficulty_dict
        self.n_days = n_days
        self.n_candidates = n_candidates
        self.n_fittest = n_fittest
        self.n_iter = n_iter
        self.p_mutation = p_mutation

    def easiness_score(self, df: np.ndarray) -> int:
        penalty = 0
        for day in self.difficulty_dict(df).T:
            easy_count = 0
            for difficulty in day:
                if difficulty == "Mudah":
                    easy_count += 1
            if easy_count < 1:
                penalty += 1
        return penalty

    def hardness_score(self, df: np.ndarray) -> int:
        penalty = 0
        for day in self.difficulty_dict(df).T:
            hard_count = 0
            for difficulty in day:
                if difficulty == "Sulit":
                    hard_count += 1
            if hard_count > 1:
                penalty += 1
        return penalty

    def duplicate_score(self, df: np.ndarray, is_per_day: bool = True) -> int:
        penalty = 0
        if is_per_day:
            df = df.T
        for time in df:
            prev_menu = ""
            for menu in time:
                if prev_menu == menu:
                    penalty += 1
                prev_menu = menu
        return penalty

    def repeated_menu_score(self, df: np.ndarray) -> int:
        repetition = pd.Series(df.flatten()).value_counts()
        penalty = repetition.gt(2).sum()  # repeated too many times
        penalty += int(repetition.eq(0).all())  # no repetitions
        return penalty

    def compute_fitness_score(self, df: np.ndarray) -> int:
        return (
            self.easiness_score(df)
            + self.hardness_score(df)
            + self.duplicate_score(df, is_per_day=False)
            + self.duplicate_score(df, is_per_day=True)
            + self.repeated_menu_score(df)
        )

    def randomise(self) -> np.ndarray:
        return np.array([
            self.menu_df.query('waktu == "Pagi"').sample(n=self.n_days).menu_id.values,
            self.menu_df.query('waktu == "Siang"').sample(n=self.n_days).menu_id.values,
            self.menu_df.query('waktu == "Malam"').sample(n=self.n_days).menu_id.values,
        ])

    def crossover(self, df_1: np.ndarray, df_2: np.ndarray) -> np.ndarray:
        mask = np.random.randint(2, size=df_1.shape).astype(bool)
        return df_1 * mask + df_2 * ~mask

    def mutate(self, candidate: np.ndarray) -> np.ndarray:
        random_menu = self.randomise()
        mask = np.random.random(size=(3, self.n_days)) < self.p_mutation
        return random_menu * mask + candidate * ~mask

    def run(self) -> pd.DataFrame:
        # Generate initial population
        candidates = [self.randomise() for _ in range(self.n_candidates)]
        scores_history = []
        for i in trange(self.n_iter):
            # Compute fitness scores and select
            fitness_scores = [
                self.compute_fitness_score(candidate) for candidate in candidates
            ]
            scores_history.append(
                {"mean": np.mean(fitness_scores), "std": np.std(fitness_scores, ddof=1)}
            )
            try:
                fitness_scores.index(0)
                break  # candidates[best_candidate_index]
            except:
                logging.info("No suitable candidates yet, continuing reproduction...")
            fittest_candidates = list(
                np.array(candidates)[np.argsort(fitness_scores)][: self.n_fittest]
            )

            # Crossover and mutation
            for (parent_1, parent_2) in combinations(fittest_candidates, 2):
                offspring = self.crossover(parent_1, parent_2)
                offspring = self.mutate(offspring)
                fittest_candidates.append(offspring)

            candidates = fittest_candidates.copy()

        fittest_candidates = list(
            np.array(candidates)[np.argsort(fitness_scores)][: self.n_fittest]
        )
        self.score_history_ = pd.DataFrame(scores_history)
        return pd.DataFrame(fittest_candidates[0])


def main():
    base_url = "https://docs.google.com/spreadsheets/d/1Uz-k2SuEuEut6YEiVfshI4kQFYLxEFvMXGcEe32bCGo/pub?gid={}&single=true&output=csv"
    difficulty_df = pd.read_csv(
        base_url.format("416476293"), skiprows=1, names=["menu_id", "difficulty"]
    )
    ingredient_df = pd.read_csv(
        base_url.format("610644228"), skiprows=1, names=["menu_id", "ingredient"]
    )
    category_df = pd.read_csv(
        base_url.format("1564738075"), skiprows=1, names=["menu_id", "category"]
    )
    menu_df = pd.read_csv(
        base_url.format("828676903"), skiprows=1, names=["menu_id", "waktu"]
    )
    difficulty_dict = np.vectorize(
        difficulty_df.set_index("menu_id").difficulty.to_dict().get
    )

    n_days = 14
    n_candidates = 10
    n_fittest = 5
    n_iter = 100
    p_mutation = 0.0

    solver = MenuSolver(
        menu_df, difficulty_dict, n_days, n_candidates, n_fittest, n_iter, p_mutation
    )
    print(solver.run())


if __name__ == "__main__":
    main()
