from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import math
import csv
import datetime
import os
from typing import Dict, List, Optional
import asyncio  # 非同期処理のため追加

# --- アプリケーションの初期化 ---
app = FastAPI(
    title="Number Expression & Decomposition API",
    description="与えられた数字の桁で自然数を表現し、その式を使って目標値を分解するAPI",
    version="1.0.0",
)

# 静的ファイルの提供設定
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ルートURL ("/") にアクセスがあったときに index.html を返すエンドポイントを明示的に定義
@app.get("/", response_class=HTMLResponse, summary="Welcome Page")
async def read_root():
    try:
        with open(os.path.join("frontend", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="index.html not found in frontend directory."
        )


# CSVファイル保存用の一時ディレクトリ（処理後に削除するため）
TEMP_DATA_DIR = "temp_data"  # 新しい一時ディレクトリ名
os.makedirs(TEMP_DATA_DIR, exist_ok=True)
generated_expressions_cache: Dict[str, Dict[int, str]] = {}

# 今回は source_number_str が変わらなければ、直前の decomposition_result を保持する形式にする
last_decomposition_result_for_comparison: Dict[str, "DecompositionResult"] = {}


# --- ヘルパー関数群 ---
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_expressions_logic(target_number_str: str) -> Dict[int, str]:
    digits_str = list(target_number_str)
    target_int = int(target_number_str)

    operators = ["+", "-", "*", "/"]
    all_possible_expressions = set()

    def generate_all_combinations(
        current_group_idx: int, current_num_groups: List[str]
    ):
        if current_group_idx == len(digits_str):
            _insert_operators_between_groups(
                0, current_num_groups[0], current_num_groups[0], current_num_groups
            )
            return

        generate_all_combinations(
            current_group_idx + 1, current_num_groups + [digits_str[current_group_idx]]
        )

        if current_num_groups:
            if len(current_num_groups[-1] + digits_str[current_group_idx]) <= 9:
                temp_groups = list(current_num_groups)
                temp_groups[-1] += digits_str[current_group_idx]
                generate_all_combinations(current_group_idx + 1, temp_groups)

    def _insert_operators_between_groups(
        op_idx: int, current_exp_str: str, first_group_str: str, num_groups: List[str]
    ):
        if op_idx == len(num_groups) - 1:
            try:
                possible_expressions = [current_exp_str]
                if int(first_group_str) != 0:
                    possible_expressions.append("-" + current_exp_str)

                for exp_str_to_eval in possible_expressions:
                    val = eval(exp_str_to_eval)
                    MAX_ALLOWABLE_RESULT = max(target_int * 20, 1000000)
                    if abs(val) > MAX_ALLOWABLE_RESULT:
                        continue
                    if isinstance(val, (int, float)) and val > 0 and val == int(val):
                        all_possible_expressions.add((int(val), exp_str_to_eval))
            except (
                SyntaxError,
                ZeroDivisionError,
                TypeError,
                OverflowError,
                ValueError,
            ):
                pass
            return

        next_group_str = num_groups[op_idx + 1]
        for op_symbol in operators:
            _insert_operators_between_groups(
                op_idx + 1,
                current_exp_str + op_symbol + next_group_str,
                first_group_str,
                num_groups,
            )

    generate_all_combinations(1, [digits_str[0]])

    unique_found_expressions = {}
    for val, exp_str in all_possible_expressions:
        if val not in unique_found_expressions:
            unique_found_expressions[val] = exp_str

    target_numbers_to_find = []
    if target_int > 1:
        target_numbers_to_find.append(1)

    for num in range(1, target_int):
        if num > 1 and is_prime(num):
            target_numbers_to_find.append(num)

    target_numbers_to_find.sort()

    final_filtered_expressions = {}
    for num_to_find in target_numbers_to_find:
        if num_to_find in unique_found_expressions:
            final_filtered_expressions[num_to_find] = unique_found_expressions[
                num_to_find
            ]
        else:
            final_filtered_expressions[num_to_find] = "見つかりませんでした"

    return final_filtered_expressions


def load_expressions_from_csv_logic(csv_filepath: str) -> Dict[int, str]:
    """
    CSVファイルから式と結果を読み込み、自然数の結果のみを辞書として返す。
    辞書は {値: 式} の形式で、値の降順にソートされる。
    """
    expressions = {}
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_filepath}")

    with open(csv_filepath, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        try:
            header = next(reader)  # ヘッダー行をスキップ
        except StopIteration:
            raise ValueError("CSVファイルが空です。")

        for row in reader:
            if len(row) == 2:
                try:
                    value = int(row[0])
                    expression = row[1]
                    if value > 0 and expression != "見つかりませんでした":
                        expressions[value] = expression
                except ValueError:
                    continue  # 数値変換エラーはスキップ

    sorted_expressions = dict(
        sorted(expressions.items(), key=lambda item: item[0], reverse=True)
    )
    return sorted_expressions


import re
from typing import List, Optional


def safe_eval_expression(expression: str, allowed_digits: List[str]) -> Optional[int]:
    """
    ユーザーが入力した式を安全に評価し、正の整数を返す。
    セキュリティのため、組み込み関数へのアクセスを制限し、不正な文字をチェックする。
    さらに、式に含まれる数字がallowed_digitsのみで構成されているかを簡易的にチェックする。
    """

    # 許可する文字: 数字、四則演算子、カッコ、空白
    base_allowed_chars_pattern = r"^[\d\s\+\-\*\/\(\)]*$"
    if not re.fullmatch(base_allowed_chars_pattern, expression):
        print(f"安全でない文字が含まれています (基本文字以外): {expression}")
        return None

    # ここから、ユーザーが入力した式に含まれる「数字文字」が、
    # allowed_digits (例: ['1', '4', '2', '8', '5', '7']) に含まれるかチェックする
    # これは「142857」という数字を使って「3」という桁を使う式は許さない、という簡易的なチェック。
    # 例: source="12", expression="1+3" -> False (3がallowed_digitsにない)
    # 例: source="12", expression="1+1" -> True (1がallowed_digitsにある)
    # 例: source="12", expression="12+1" -> True (1,2がallowed_digitsにある)

    # 式の中からすべての数字の塊を抽出
    all_numbers_in_expression = re.findall(r"\d+", expression)

    # 各数字の塊を個々の桁に分解し、それらの桁が allowed_digits に含まれているかチェック
    for num_str in all_numbers_in_expression:
        for digit_char in num_str:
            if digit_char not in allowed_digits:
                print(
                    f"許可されていない桁 '{digit_char}' が式に含まれています: {expression}"
                )
                return None

    try:
        # eval() をより安全に実行するために、__builtins__ を空にする
        result = eval(expression, {"__builtins__": None}, {})

        # 結果が整数で、かつ正の値であることを確認
        if isinstance(result, (int, float)) and result > 0 and result == int(result):
            return int(result)

        print(f"評価結果が正の整数ではありません: {result}")
        return None
    except (
        SyntaxError,
        ZeroDivisionError,
        TypeError,
        NameError,
        OverflowError,
        ValueError,
    ) as e:
        print(f"式の評価エラー: {e} for expression: {expression}")
        return None


def calculate_expression_metrics(expression: str) -> Dict[str, int]:
    """
    式の長さと演算子数を計算する。
    """
    stripped_expression = expression.replace(" ", "")  # スペース除去
    length = len(stripped_expression)
    operator_count = sum(stripped_expression.count(op) for op in ["+", "-", "*", "/"])
    return {"length": length, "operator_count": operator_count}


class DecompositionStep(BaseModel):
    expression: str
    value_used: int
    remaining_before: int
    remaining_after: int


class DecompositionResult(BaseModel):
    decomposition_steps: List[DecompositionStep]
    final_remaining: int
    total_expressed_value: int
    full_expression_string: str
    status: str
    auto_gen_expression_length: Optional[int] = None  # 追加
    auto_gen_operator_count: Optional[int] = None  # 追加


def express_target_value_logic(
    target_value: int, available_expressions: Dict[int, str]
) -> DecompositionResult:
    """
    目標値を、利用可能な式を使って表現する。
    貪欲法で、残りの値から最も大きい自然数を繰り返し見つけて引く。
    （FastAPIのエンドポイントから呼び出されるコアロジック）
    """
    remaining_value = target_value
    decomposition_steps = []
    total_expressed = 0

    while remaining_value > 0:
        found_match = False
        best_match_value = 0
        best_match_expression = ""

        for value, expression in available_expressions.items():
            if value <= remaining_value:
                best_match_value = value
                best_match_expression = expression
                found_match = True
                break

        if found_match:
            step = DecompositionStep(
                expression=best_match_expression,
                value_used=best_match_value,
                remaining_before=remaining_value,
                remaining_after=remaining_value - best_match_value,
            )
            decomposition_steps.append(step)
            remaining_value -= best_match_value
            total_expressed += best_match_value
        else:
            step = DecompositionStep(
                expression="No further decomposition",
                value_used=0,
                remaining_before=remaining_value,
                remaining_after=remaining_value,
            )
            decomposition_steps.append(step)
            break

    full_expression_parts = [
        step.expression
        for step in decomposition_steps
        if step.expression != "No further decomposition"
    ]
    full_expression_string = (
        " + ".join(full_expression_parts) if full_expression_parts else ""
    )

    status_message = (
        "完全に表現されました。"
        if remaining_value == 0
        else "完全に表現できませんでした。"
    )

    # メトリクス計算を追加
    auto_gen_expression_length = None
    auto_gen_operator_count = None
    if full_expression_string:
        metrics = calculate_expression_metrics(full_expression_string)
        auto_gen_expression_length = metrics["length"]
        auto_gen_operator_count = metrics["operator_count"]

    return DecompositionResult(
        decomposition_steps=decomposition_steps,
        final_remaining=remaining_value,
        total_expressed_value=total_expressed,
        full_expression_string=full_expression_string,
        status=status_message,
        auto_gen_expression_length=auto_gen_expression_length,
        auto_gen_operator_count=auto_gen_operator_count,
    )


# FastAPI エンドポイント


class ProcessNumberRequest(BaseModel):
    source_number_str: str  # 式を生成するための元の数字 (例: "142857")
    target_value: int  # 分解したい目標値 (例: 27)


# ユーザー入力式の評価リクエスト用モデル
class EvaluateUserExpressionRequest(BaseModel):
    source_number_str: str  # 元の数字（例: "142857"） - 比較のために必要
    target_value: int  # 分解したい目標値（例: 27）
    user_expression: str  # ユーザーが入力した式文字列（例: "1+4*2-8"）


class EvaluateUserExpressionResponse(BaseModel):
    user_expression: str  # ユーザーが入力した式
    user_expression_value: Optional[int]
    # ユーザーの式が評価された値 (評価できない場合はNone)
    user_expression_length: int  # ユーザーの式の文字数（スペース除く）
    user_operator_count: int  # ユーザーの式の演算子数
    is_valid_expression: bool  # 式が有効なPythonの式として評価できたか
    is_correct_value: bool  # ユーザーの式の評価値が目標値と一致したか
    comparison_result: str  # 自動生成された式との比較結果のメッセージ
    auto_gen_expression: Optional[str] = None  # 自動生成された分解の完全な式（比較用）
    auto_gen_expression_length: Optional[int] = None  # 自動生成された式の文字数
    auto_gen_operator_count: Optional[int] = None  # 自動生成された式の演算子数


@app.post(
    "/process_number",
    response_model=DecompositionResult,  # 分解結果を返す
    summary="与えられた数字の桁で式を生成し、その式を使って目標値を分解します",
)
async def process_number(request: ProcessNumberRequest):
    """
    元の数字の桁から式を生成し、その式を使って目標値を分解する。
    """
    # 1. source_number_str のバリデーション
    try:
        int(request.source_number_str)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="source_number_strは有効な数字である必要があります。",
        )
    if request.target_value <= 0:
        raise HTTPException(
            status_code=400, detail="target_valueは正の整数である必要があります。"
        )

    # 2. 式の生成と一時CSVファイルへの書き出し
    print(f"--- 調査開始 ({request.source_number_str}) ---")
    print(
        f"'{request.source_number_str}' の桁を固定順序で使い、先頭にマイナスを付けて1と、1から{int(request.source_number_str)-1}までの素数を表現できるか調査中..."
    )

    results_from_generation = generate_expressions_logic(request.source_number_str)

    # 一時CSVファイル名を作成
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    temp_csv_filename = f"temp_results_{request.source_number_str}_{timestamp}.csv"
    temp_csv_filepath_full = os.path.join(TEMP_DATA_DIR, temp_csv_filename)

    try:
        with open(temp_csv_filepath_full, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Target Number", "Expression Found"])
            for num, expr in results_from_generation.items():
                csv_writer.writerow([num, expr])

        print(f"一時結果を '{temp_csv_filepath_full}' に書き出しました。")

        # 3. 生成したCSVファイルを読み込む
        available_expressions_for_decomposition = load_expressions_from_csv_logic(
            temp_csv_filepath_full
        )
        if not available_expressions_for_decomposition:
            raise HTTPException(
                status_code=400,
                detail="生成された式から有効な分解用式が見つかりませんでした。",
            )

        # 4. 目標値を分解する
        decomposition_result = express_target_value_logic(
            request.target_value, available_expressions_for_decomposition
        )
        last_decomposition_result_for_comparison[request.source_number_str] = (
            decomposition_result
        )

        return decomposition_result

    except Exception as e:
        # エラー発生時は一時ファイルを削除し、HTTPExceptionを返す
        print(f"処理中にエラーが発生しました: {e}")
        raise HTTPException(
            status_code=500, detail=f"処理中にエラーが発生しました: {e}"
        )
    finally:
        # 5. 処理終了後、一時ファイルを削除
        if os.path.exists(temp_csv_filepath_full):
            os.remove(temp_csv_filepath_full)
            print(f"一時ファイル '{temp_csv_filepath_full}' を削除しました。")


@app.post(
    "/evaluate_user_expression",
    response_model=EvaluateUserExpressionResponse,
    summary="ユーザーが入力した式を評価し、自動生成された式と比較します",
)
async def evaluate_user_expression(request: EvaluateUserExpressionRequest):
    """
    ユーザーが入力した式を評価し、その値、長さ、計算量を計算し、
    自動生成された式の情報と比較して結果を返します。
    """
    user_expression_str = request.user_expression.replace(" ", "")  # スペースを除去
    user_metrics = calculate_expression_metrics(user_expression_str)
    user_expression_length = user_metrics["length"]
    user_operator_count = user_metrics["operator_count"]

    user_expression_value = safe_eval_expression(
        request.user_expression, list(request.source_number_str)
    )

    is_valid_expression = user_expression_value is not None
    is_correct_value = False
    if is_valid_expression and user_expression_value == request.target_value:
        is_correct_value = True

    comparison_result = "きみの式は評価できませんでした。"  # 初期値

    # 自動生成された式の情報を取得
    auto_gen_expression = None
    auto_gen_expression_length = None
    auto_gen_operator_count = None

    # 前回 /process_number で計算された分解結果をキャッシュから取得
    # source_number_str に対応する DecompositionResult が存在するかチェック
    if request.source_number_str in last_decomposition_result_for_comparison:
        cached_result = last_decomposition_result_for_comparison[
            request.source_number_str
        ]

        # キャッシュされた結果が、今回ユーザーが挑戦している target_value と一致するか確認
        # もし異なる target_value であれば、比較は意味をなさないため無視
        if (
            cached_result.total_expressed_value == request.target_value
            and cached_result.final_remaining == 0
        ):
            # 完全に表現できた場合のみ比較対象とする
            auto_gen_expression = cached_result.full_expression_string
            auto_gen_expression_length = cached_result.auto_gen_expression_length
            auto_gen_operator_count = cached_result.auto_gen_operator_count

            # ユーザーの式が正しい値に評価できた場合のみ比較
            if is_correct_value:
                if (
                    auto_gen_expression_length is not None
                    and user_expression_length < auto_gen_expression_length
                ):
                    comparison_result = f"きみの式の方が自動生成された式より {auto_gen_expression_length - user_expression_length} 文字短いよ！素晴らしい！"
                elif (
                    auto_gen_expression_length is not None
                    and user_expression_length == auto_gen_expression_length
                ):
                    # 長さが同じ場合、演算子数で比較するロジック
                    if (
                        auto_gen_operator_count is not None
                        and user_operator_count < auto_gen_operator_count
                    ):
                        comparison_result = f"きみの式は自動生成された式と同じ長さだけど、演算子が {auto_gen_operator_count - user_operator_count} 個少ないよ！すごい！"
                    elif (
                        auto_gen_operator_count is not None
                        and user_operator_count == auto_gen_operator_count
                    ):
                        comparison_result = "きみの式は自動生成された式と同じ長さで、同じ演算子数だよ！良くできたね！"
                    else:
                        comparison_result = "きみの式は自動生成された式と同じ長さだけど、より多くの演算子を使っているようだね。もっとシンプルにできるかも？"
                else:
                    if auto_gen_expression_length is not None:
                        comparison_result = f"きみの式は自動生成された式より {user_expression_length - auto_gen_expression_length} 文字長いよ。"
                    else:
                        comparison_result = "自動生成された式と比べることができませんでした。再計算してみてください。"
            else:
                comparison_result = (
                    "きみの式は目標値に分解できませんでした。再度挑戦してね！"
                )
        else:
            comparison_result = "自動生成された式は、目標値に完全に分解できませんでした。そのため比較ができません。"
    else:
        comparison_result = (
            "まず「計算と分解を実行」して、自動生成された式を準備してください。"
        )

    return EvaluateUserExpressionResponse(
        user_expression=request.user_expression,
        user_expression_value=user_expression_value,
        user_expression_length=user_expression_length,
        user_operator_count=user_operator_count,
        is_valid_expression=is_valid_expression,
        is_correct_value=is_correct_value,
        comparison_result=comparison_result,
        auto_gen_expression=auto_gen_expression,
        auto_gen_expression_length=auto_gen_expression_length,
        auto_gen_operator_count=auto_gen_operator_count,
    )


# uvicorn main:app --reload --port 8000
# http://localhost:8000
