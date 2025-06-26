import pandas as pd


def styler(df: pd.DataFrame, subset: list, a_color: str, b_color: str) -> pd.DataFrame.style:
    fn = lambda x: f"background-color: {(a_color, b_color)[x < 0]}; color:black"  # noqa: E731
    return df.style.map(func=fn, subset=pd.IndexSlice[slice(None), subset])
