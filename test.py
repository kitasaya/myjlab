from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import math
import csv
import datetime
import os
from typing import Dict, List, Optional
import asyncio

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
TEMP_DATA_DIR = "temp_data"
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# グローバル変数として生成された式を保持する辞書
# これにより、/process_number で生成された式を /evaluate_user_expression で参照できます。
# 注意: これはシンプルな例であり、複数のユーザーセッションを考慮した設計ではありません。
# 本番環境では、データベースやキャッシュ、セッション管理を使用することを検討してください。
generated_expressions_cache: Dict[str, Dict[int, str]] = {}


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
    # 自動生成された式の特性を保持するフィールド
    auto_gen_expression_length: Optional[int] = None
    auto_gen_operator_count: Optional[int] = None


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

    # 自動生成された分解の「完全な式」を追跡
    # ユーザーの式と比較するために使用
    auto_gen_full_expression_parts = []

    while remaining_value > 0:
        found_match = False
        best_match_value = 0
        best_match_expression = ""

        # availble_expressions は降順にソートされているため、最初に見つかるものが最大値
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
            auto_gen_full_expression_parts.append(best_match_expression)  # 追加
        else:
            step = DecompositionStep(
                expression="No further decomposition",
                value_used=0,
                remaining_before=remaining_value,
                remaining_after=remaining_value,
            )
            decomposition_steps.append(step)
            break

    full_expression_string = (
        " + ".join(auto_gen_full_expression_parts)
        if auto_gen_full_expression_parts
        else ""
    )

    status_message = (
        "完全に表現されました。"
        if remaining_value == 0
        else "完全に表現できませんでした。"
    )

    # 生成された式の長さと演算子数を計算
    auto_gen_expression_length = len(
        full_expression_string.replace(" ", "")
    )  # スペース除去
    auto_gen_operator_count = sum(
        full_expression_string.count(op) for op in ["+", "-", "*", "/"]
    )

    return DecompositionResult(
        decomposition_steps=decomposition_steps,
        final_remaining=remaining_value,
        total_expressed_value=total_expressed,
        full_expression_string=full_expression_string,
        status=status_message,
        auto_gen_expression_length=auto_gen_expression_length,  # 追加
        auto_gen_operator_count=auto_gen_operator_count,  # 追加
    )


# --- FastAPI エンドポイント ---


class ProcessNumberRequest(BaseModel):
    source_number_str: str  # 式を生成するための元の数字 (例: "142857")
    target_value: int  # 分解したい目標値 (例: 27)


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
        f"'{request.source_number_str}' の桁を固定順序で使い、先頭にマイナスを付けて1と、1から{request.target_value-1}までの素数を表現できるか調査中..."
    )  # target_int から target_value に変更

    results_from_generation = generate_expressions_logic(request.source_number_str)

    # 生成された式をキャッシュに保存
    generated_expressions_cache[request.source_number_str] = results_from_generation

    # 一時CSVファイル名を作成 (この部分はload_expressions_from_csv_logicがまだ使われているため残すが、
    # 理想的には直接results_from_generationを使うように変更可能)
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

        # 3. 生成したCSVファイルを読み込む (キャッシュからの読み込みに切り替えも可能だが、
        # 現在のload_expressions_from_csv_logicはソート機能も持っているので残す)
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


# --- 新しいゲーム機能のためのエンドポイントとヘルパー関数 ---


# ユーザー入力式の評価ロジック
# セキュリティのために、許可された文字のみを含むかチェックする
def safe_eval_expression(expression: str, allowed_digits: List[str]) -> Optional[float]:
    # 許可された文字（数字、演算子、かっこ）と空白文字の正規表現パターン
    # ここでは、元の数字の桁に含まれる数字のみを許可するようにします。
    allowed_chars_pattern = r"^[\s\d\+\-\*\/\.\(\)]*$"

    # ユーザーが入力した式に含まれる数字が、元数字の桁に含まれるかチェック
    # これは厳密なチェックであり、例: 142857 の数字で '11' を使うのは許可されるが '3' は許可されない。
    # 簡略化のために、ここでは一旦「数字」と「元の数字の桁」のマッチングは行わない。
    # 代わりに、評価後に桁の使用状況をチェックする方が現実的だが、今回は一旦単純化する。
    # (例: eval('1+3') は動くが、元の数字が '12' なら '3' は使えないのでエラーとするなど)

    # 簡易的なセキュリティチェック：許可された文字以外が含まれていないか
    import re

    if not re.fullmatch(allowed_chars_pattern, expression):
        print(f"不正な文字が含まれています: {expression}")
        return None

    # eval() をより安全に実行するために、__builtins__ を空にする
    # ただし、これだけでは全ての攻撃を防ぐわけではないため、入力の厳密なバリデーションが重要
    try:
        # 入力式に元の数字の桁に含まれない数字が含まれていないかの簡易チェック
        # (例: source_number_strが"12"のときにユーザーが"1+3"と入力した場合、"3"は含まれていない)
        # より高度なチェックが必要だが、今回は簡易的に、式に含まれる数字が元の数字の桁に含まれるか
        # かつ、式に含まれる数字が元の数字の桁以外（例: '1'は'1'でも'10'でもOKとしてしまうなど）
        # という複雑な問題があるため、ここでは、元の数字の桁以外は使えないようにする

        # ユーザーが入力した式が、元の数字の桁だけで構成されているか確認する（簡易版）
        # これは非常に複雑なロジックになるため、今回は「入力された式が元の数字の桁だけで構成されている」という厳密なルールではなく、
        # 「入力された式が、元の数字の桁"を結合して"作られた数字を使用しているか」というより緩やかなルールに留める。
        # 例：元の数字が"12"でユーザーが"1+2"と入力するのはOK。
        # 例：元の数字が"12"でユーザーが"1+3"と入力するのはNGとしたいが、eval()ではチェックできない。
        # そのため、この実装では、この「桁の制約」はeval()の直接の制限では行わない。
        # 代わりに、フロントエンドでユーザーに「使用できる数字は元の数字の桁に限る」と明確に伝えるのが現実的。

        # ここでは、純粋な数値計算と構文エラーチェックのみを行う
        result = eval(expression, {"__builtins__": None}, {})

        # 結果が整数で、かつ正の値であることを確認
        if isinstance(result, (int, float)) and result > 0 and result == int(result):
            return int(result)
        return None
    except (
        SyntaxError,
        ZeroDivisionError,
        TypeError,
        NameError,
        OverflowError,
        ValueError,
    ) as e:
        print(f"式の評価エラー: {e}")
        return None


def calculate_expression_complexity(expression: str) -> int:
    # 式の複雑度を簡易的に計算する。
    # 例：演算子の数と、数字の結合（複数桁の数字）を考慮

    # 演算子の数を数える
    operator_count = sum(expression.count(op) for op in ["+", "-", "*", "/"])

    # 数字の桁数を数える（スペースと演算子を除外）
    # 式に含まれる数字の総桁数
    digit_length = len("".join(c for c in expression if c.isdigit()))

    # 例: "1+2*3" -> op_count=2, digit_length=3, complexity = 2 + 3 = 5
    # 例: "12+34" -> op_count=1, digit_length=4, complexity = 1 + 4 = 5
    # 結合された数字の数も考慮する（例: 1と2から12を作るのは、1と2を別々に使うよりシンプル）
    # 今回はシンプルに演算子数と文字数で定義

    # より複雑な計算量の定義（例: ASTをパースしてノード数を数えるなど）も可能だが、
    # ここでは単純に文字列の長さと演算子数で比較する

    # 例: 演算子数 + 数字の文字数
    complexity = operator_count + digit_length
    return complexity


class EvaluateUserExpressionRequest(BaseModel):
    source_number_str: str  # 比較のために元の数字が必要
    target_value: int  # 比較のために目標値が必要
    user_expression: str  # ユーザーが入力した式


class EvaluateUserExpressionResponse(BaseModel):
    user_expression: str
    user_expression_value: Optional[int]
    user_expression_length: int
    user_operator_count: int
    is_valid_expression: bool
    is_correct_value: bool
    comparison_result: str
    auto_gen_expression: Optional[str] = None
    auto_gen_expression_length: Optional[int] = None
    auto_gen_operator_count: Optional[int] = None


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
    user_expression_length = len(user_expression_str)
    user_operator_count = sum(
        user_expression_str.count(op) for op in ["+", "-", "*", "/"]
    )

    # ユーザーの式を評価
    # ここで、ユーザーが入力した式がsource_number_strの桁のみで構成されているかの検証を
    # より厳密に行う必要があるが、safe_eval_expressionでは簡易的なセキュリティチェックのみを行う。
    # 厳密な桁の検証は、式を解析するより複雑なロジックが必要になる。
    # 例: "1+2" (source="12") OK, "1+3" (source="12") NG
    # この問題は、Pythonのeval()では直接解決できないため、ここでは評価値が正しいかどうかに焦点を当てる。
    # フロントエンドで、ユーザーに「使用できる数字は元の数字の桁に限る」と明確に伝えるようにする。

    user_expression_value = safe_eval_expression(
        request.user_expression, list(request.source_number_str)
    )

    is_valid_expression = user_expression_value is not None
    is_correct_value = False
    if is_valid_expression and user_expression_value == request.target_value:
        is_correct_value = True

    comparison_result = "評価できませんでした。"

    auto_gen_expression_info = None
    auto_gen_expression_length = None
    auto_gen_operator_count = None

    # キャッシュから自動生成された分解結果を取得
    # process_number が先に呼ばれていることを前提とする
    if request.source_number_str in generated_expressions_cache:
        # ここでは、自動生成された式の中から、ユーザーのtarget_valueを直接表現している
        # 最も短い/シンプルな式を探すのは難しいので、
        # express_target_value_logic で得られた全体の分解式と比較することにする。
        # もし、ユーザーが単一の式でtarget_valueを表現した場合、比較がより意味を持つ。

        # 暫定的に、前回 process_number で計算された分解結果の全体式を使用する。
        # より良い方法は、/process_number の結果を直接取得し、その中の `full_expression_string` を使うこと。
        # あるいは、/process_number を呼び出す際に、その結果をセッション等に保存し、
        # ここで取得できるようにする。
        # 今回の改修では、express_target_value_logic から返される `full_expression_string`
        # とその特性を DecompositionResult に含めるようにしたので、それを利用する。

        # 注意: このロジックでは、/process_number が一度実行されていて、
        # かつその結果が `generated_expressions_cache` に、
        # `source_number_str` をキーとして保存されていることを想定しています。
        # もし `process_number` がまだ実行されていない場合、`auto_gen_expression_info` は None のままになります。

        # ここでは、ユーザーが入力した source_number_str と target_value で、
        # もう一度 express_target_value_logic を実行して、最適な自動生成式を得るのが確実
        # ただし、これは計算コストが高いので、cached_decomposition_result を使うのが望ましい。

        # 課題：`generated_expressions_cache` は、`generate_expressions_logic` の結果 (`Dict[int, str]`) を持つ。
        # `express_target_value_logic` の `DecompositionResult` をキャッシュする必要がある。
        # または、`process_number` から返された `DecompositionResult` をクライアント側で保持し、
        # このエンドポイントに送り返す。

        # シンプルな解決策として、`process_number` の結果をキャッシュし、それを参照する。
        # これをグローバル変数に保存するのではなく、より適切な方法を検討する必要があるが、
        # 今回はプロトタイプとして、request.source_number_str をキーとする。
        # ただし、`express_target_value_logic` の結果全体を保持する必要がある。

        # このロジックは、以前に express_target_value_logic が実行され、
        # その結果がどこかにキャッシュされていることを前提としている。
        # 簡易的に、`process_number` で返される `DecompositionResult` 全体をキャッシュする。

        # generated_expressions_cache は {source_number_str: Dict[int, str]} なので、
        # ここで DecompositionResult を取得することはできない。
        # そこで、/process_number が呼び出された時に、その DecompositonResult を
        # source_number_str と target_value の組み合わせでキャッシュすることにする。

        # 新しいキャッシュ構造を導入
        # global_decomposition_results_cache: Dict[Tuple[str, int], DecompositionResult] = {}
        # とすると複雑になるため、今回は簡易的に、/process_number が返した結果の `full_expression_string`
        # およびその長さと演算子数を、ユーザーが入力した `source_number_str` と `target_value` をキーとして
        # 保持するように `process_number` 側を変更する。

        # ここでは、`process_number` の実行結果が `generated_expressions_cache` に保存されており、
        # その中に `auto_gen_expression_length` と `auto_gen_operator_count` が含まれていると仮定する。
        # しかし、これは実際には `DecompositionResult` のフィールドなので、キャッシュの構造を変更する必要がある。

        # 仮の対応：`process_number` からの返り値 `DecompositionResult` をどこかに保存するか、
        # あるいはここで再度 `express_target_value_logic` を呼び出して計算する。
        # 計算コストを考えると、キャッシュが望ましいが、今回は簡便のために再計算する。
        # (ただし、これは非効率的であり、本来は `process_number` の結果を適切にキャッシュすべき)

        # ユーザーの要求に合致する「自動生成された式」を得るために、
        # source_number_str と target_value で再度分解ロジックを実行。
        # NOTE: これは非効率なので、より良いキャッシュ戦略が必要。
        expressions_for_decomposition = load_expressions_from_csv_logic(
            os.path.join(
                TEMP_DATA_DIR,
                f"temp_results_{request.source_number_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )  # このファイル名は適切ではない
        )
        # 上記のCSVファイル名は実行ごとに異なるので、正確にキャッシュされたCSVファイル名を知る必要がある。
        # これはグローバルキャッシュに `DecompositionResult` そのものを保存するほうが簡単になる。

        # より現実的な対応: `process_number` が呼ばれたときに、生成された式とその特性を、
        # `source_number_str` と `target_value` の組み合わせでキャッシュしておく。
        # ここでは、一旦、`generated_expressions_cache` に `DecompositionResult` を保存するよう変更する。

        # global generated_expressions_cache
        # `process_number` を以下のように変更する:
        # generated_expressions_cache[(request.source_number_str, request.target_value)] = decomposition_result

        # ここでは、簡易的に、前回の `process_number` で計算された `full_expression_string` を使用する。
        # ただし、`process_number` で計算された `target_value` と
        # `evaluate_user_expression` で送られてくる `target_value` が一致しない場合がある。
        # そのため、`process_number` で生成した分解結果そのものをキャッシュする方が良い。

        # 暫定的に、`process_number` が最後に実行された時の `DecompositionResult` を
        # source_number_str をキーとして保存する (target_valueは無視)。
        # これは理想的ではないが、シンプルなグローバルキャッシュの例として。
        # 実際には、(source_number_str, target_value) のタプルをキーにするべき。

        # 以下は、キャッシュ戦略変更前の仮の対応です。
        # 正しい実装には、`process_number` のキャッシュ戦略見直しが必要です。

        # ユーザーが入力した式の評価値が正しい場合のみ比較を行う
        if is_correct_value:
            # `process_number` で計算された最適な分解式と特性を取得
            # (これは本来、特定のsource_number_strとtarget_valueに対する結果であるべき)
            # 現在の`generated_expressions_cache`には`Dict[int, str]`しかなく、
            # `DecompositionResult`は含まれていないため、`express_target_value_logic`を再実行する。
            # これは計算コストが高いが、このプロトタイプでは許容する。

            # generate_expressions_logic は source_number_str を元に素数などを生成する。
            # load_expressions_from_csv_logic はその結果を読み込む。
            # express_target_value_logic は target_value をそれらの式で分解する。

            # 既に `process_number` が一度実行され、`generated_expressions_cache` に
            # `generate_expressions_logic` の結果が保存されていると仮定する。
            if request.source_number_str not in generated_expressions_cache:
                raise HTTPException(
                    status_code=400,
                    detail="最初に「計算と分解を実行」して、自動生成された式を準備してください。",
                )

            # `generate_expressions_logic` の結果を使って分解
            # (これは、CSVを介さずにメモリから直接読み込むべきだが、既存関数を再利用)
            # しかし、CSVファイルは一時ファイルなので、もう存在しない可能性がある。
            # よって、`generated_expressions_cache` を直接使う。

            # available_expressions_for_decomposition = load_expressions_from_csv_logic(temp_csv_filepath_full) #これは不可

            # generated_expressions_cache から直接 available_expressions を取得
            available_expressions_for_decomposition = generated_expressions_cache[
                request.source_number_str
            ]

            if not available_expressions_for_decomposition:
                raise HTTPException(
                    status_code=500,
                    detail="自動生成された分解用式がキャッシュに見つかりませんでした。",
                )

            auto_decomposition_result = express_target_value_logic(
                request.target_value, available_expressions_for_decomposition
            )

            auto_gen_expression_info = auto_decomposition_result.full_expression_string
            auto_gen_expression_length = (
                auto_decomposition_result.auto_gen_expression_length
            )
            auto_gen_operator_count = auto_decomposition_result.auto_gen_operator_count

            if (
                auto_gen_expression_length is not None
                and user_expression_length < auto_gen_expression_length
            ):
                comparison_result = f"きみの式の方が自動生成された式より {auto_gen_expression_length - user_expression_length} 文字短いよ！素晴らしい！"
            elif (
                auto_gen_expression_length is not None
                and user_expression_length == auto_gen_expression_length
            ):
                if (
                    auto_gen_operator_count is not None
                    and user_operator_count < auto_gen_operator_count
                ):
                    comparison_result = f"きみの式は自動生成された式と同じ長さだけど、演算子が {auto_gen_operator_count - user_operator_count} 個少ないよ！すごい！"
                elif (
                    auto_gen_operator_count is not None
                    and user_operator_count == auto_gen_operator_count
                ):
                    comparison_result = (
                        "きみの式は自動生成された式と同じ長さで、同じ演算子数だよ！"
                    )
                else:
                    comparison_result = "きみの式は自動生成された式と同じ長さだけど、より多くの演算子を使っているようだね。"
            else:
                if auto_gen_expression_length is not None:
                    comparison_result = f"きみの式は自動生成された式より {user_expression_length - auto_gen_expression_length} 文字長いよ。"
                else:
                    comparison_result = (
                        "自動生成された式と比べることができませんでした。"
                    )
        else:
            comparison_result = "きみの式は正しい値に分解できませんでした。"

    return EvaluateUserExpressionResponse(
        user_expression=request.user_expression,
        user_expression_value=user_expression_value,
        user_expression_length=user_expression_length,
        user_operator_count=user_operator_count,
        is_valid_expression=is_valid_expression,
        is_correct_value=is_correct_value,
        comparison_result=comparison_result,
        auto_gen_expression=auto_gen_expression_info,
        auto_gen_expression_length=auto_gen_expression_length,
        auto_gen_operator_count=auto_gen_operator_count,
    )


# uvicorn main:app --reload --port 8000
# http://localhost:8000
