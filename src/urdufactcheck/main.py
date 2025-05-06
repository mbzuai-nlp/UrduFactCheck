from openfactcheck import OpenFactCheck, OpenFactCheckConfig

CONFIG_PATH = "config.json"

config = OpenFactCheckConfig(CONFIG_PATH)


def callback(
    index,
    sample_name,
    solver_name,
    input_name,
    output_name,
    input,
    output,
    continue_run,
):
    print(
        f"Callback: {index}, {sample_name}, {solver_name}, {input_name}, {output_name}, {input}, {output}, {continue_run}"
    )


response = OpenFactCheck(config).ResponseEvaluator.evaluate(
    response="قائداعظم محمد علی جناح پاکستان کے بانی اور پہلے گورنر جنرل تھے۔ انہوں نے پاکستان کے قیام کے لیے جدوجہد کی اور مسلم کمیونٹی کے حقوق کے لیے آواز اٹھائی۔ انہیں “قائداعظم” یعنی “عظیم رہنما” کے لقب سے پکارا جاتا ہے۔",
    callback=callback,
)

print("Overall Result: ", response)
