from string import Template
from dataclasses import dataclass

import numpy

test_data = [('void xfrm_policy_walk_done(struct xfrm_policy_walk *walk, struct net *net)\n', 913.9556608200073),
             ('{\n', 34.87690353393555), ('if (list_empty(&walk->walk.all))\n', 532.4354400634766),
             ('return;\n', 70.820556640625), ('\n', 0),
             ('spin_lock_bh(&net->xfrm.xfrm_policy_lock); /*FIXME where is net? */\n', 876.4920692443848),
             ('list_del(&walk->walk.all);\n', 360.30310249328613),
             ('spin_unlock_bh(&net->xfrm.xfrm_policy_lock);\n', 681.8842926025391), ('}', 37.052940368652344)]

token_test_data = [[('static', 28.94988250732422), ('struct', 24.799318313598633), ('sk_buff', 62.73761558532715),
                    ('*netlink_trim(struct', 149.1546220779419), ('sk_buff', 58.45393943786621),
                    ('*skb,', 69.57473564147949), ('gfp_t', 69.33642196655273), ('allocation)', 56.79461097717285)],
                   [('{', 21.080068588256836)], [('int', 19.19388198852539), ('delta;', 51.66716766357422)],
                   [('WARN_ON(skb->sk', 179.03374767303467), ('!=', 19.06562614440918), ('NULL);', 50.10061454772949)],
                   [('delta', 40.87398910522461), ('=', 14.088622093200684), ('skb->end', 66.70916748046875),
                    ('-', 13.966950416564941), ('skb->tail;', 98.00306987762451)],
                   [('if', 15.444297790527344), ('(is_vmalloc_addr(skb->head)', 230.27002334594727),
                    ('||', 16.08974266052246), ('delta', 17.941478729248047),
                    ('*', 10.787075996398926), ('2', 16.496219635009766), ('<', 12.199067115783691),
                    ('skb->truesize)', 138.93876266479492)],
                   [('return', 22.02549934387207), ('skb;', 52.98655891418457)],
                   [('if', 16.36863899230957), ('(skb_shared(skb))', 172.6628246307373), ('{', 13.326693534851074)],
                   [('struct', 21.880001068115234), ('sk_buff', 65.34987926483154), ('*nskb', 63.133463859558105),
                    ('=', 17.281051635742188), ('skb_clone(skb,', 190.1592311859131),
                    ('allocation);', 73.53987884521484)], [( \
        'if', 19.306476593017578), ('(!nskb)', 127.32530212402344)],
                   [('return', 27.599058151245117), ('skb;', 61.165011405944824)],
                   [('consume_skb(skb);', 203.8105001449585)],
                   [('skb', 45.20133876800537), ('=', 19.812705993652344), ('nskb;', 88.34631156921387)],
                   [('}', 21.77391242980957)],
                   [('if', 21.74557113647461), ('(!pskb_expand_head(skb,', 330.8136796951294),
                    ('0,', 53.36063766479492), ('-delta,', 105.01745986938477), ('allocation))', 83.24134826660156)],
                   [('skb->truesize', 173.08755683898926), ('-=', 30.441232681274414), ('delta;', 50.67720985412598)],
                   [('return', 31.150352478027344), ('skb;', 73.73404502868652)], [('}', 18.650012969970703)]]

@dataclass
class VisualizeInfo:
    model_name:str
    pred:bool
    label:bool
    cwe:str
    cve:str
    git_url : str
    raw_func:str

def interpolate_rgb_color(color1, color2, steps):
    r1, g1, b1 = color1
    r2, g2, b2 = color2

    # Calculate the step size for each color channel.
    step_size_r = (r2 - r1) / steps
    step_size_g = (g2 - g1) / steps
    step_size_b = (b2 - b1) / steps

    # Interpolate colors and store in a list.
    interpolated_colors = []
    for step in range(steps):
        r = int(r1 + step_size_r * step)
        g = int(g1 + step_size_g * step)
        b = int(b1 + step_size_b * step)
        interpolated_colors.append((r, g, b))

    return interpolated_colors


def rgb_to_hex(color):
    r, g, b = color
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


HeatMapColorLow = (254, 228, 145)
HeatMapColorHigh = (255, 0, 0)
HeatMapColorSteps = 50
HeatMapColors = interpolate_rgb_color(HeatMapColorLow, HeatMapColorHigh, HeatMapColorSteps)

visualize_html_template = Template(f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>$name</title>
</head>
<link href='https://fonts.googleapis.com/css?family=JetBrains Mono' rel='stylesheet'>

<style>
.container{{
  display: flex;
  padding: 0px;
}}

.code {{
    width: fit-content;
    margin:0px;
    padding-top: 4px;
    padding-bottom: 4px;
  	display: inline-block;
  	color: black;
  	background-clip: padding-box;

}}

.line_number {{
    margin-right: 10px;
  	display: inline-block;
    color: white;
    padding-top: 4px;
    padding-bottom: 4px;
    width: 40px;
    text-align: right; 
    user-select: none;
}}

.util{{
    display: flex;
    margin-top: 50px;
    margin-left: 50px;
    column-gap: 20px;
    align-items: center;
}}

div{{
    font-size: 17px;
}}

body{{
    background: rgb(58, 58, 58);
    font-family: 'JetBrains Mono';
}}

#heatmap{{
    border-radius: 4px;
    width: 600px;
    height: 60px;
    background-image: linear-gradient(to right,{rgb_to_hex(HeatMapColorLow)}, {rgb_to_hex(HeatMapColorHigh)});
}}

h1{{
    font-family: 'JetBrains Mono';
    font-size: 35px;
    color: white;
    margin-left: 20px;
}}

#vulnerable{{
    color: red;
    font-size: 25px;

}}

#non-vulnerable{{
    color: green;
    font-size: 25px;

}}

#model_output{{
    color: white;
    
}}

a , a:visited, a:link , a:hover{{
    display: block;
    color:white;
}}

</style>

<script>
function copyCode() {{
    navigator.clipboard.writeText(`$raw_func`)
}}
</script>

<body>
  <h1>$name </h1>
  $codes
  
  <div class="util">
      $vulnerable
      <a href="$git_url" target="_blank">Jump to Git URL</a>
      <a href="$cve_url" target="_blank">Jump to CVE URL</a>
      <a href="#" onclick="copyCode()">Copy Code</a>
      <div id="heatmap"/> 
  </div>
</body>

</html>
""")


def vulnerable_html_element(pred:bool , label:bool):
    pred_str = "vulnerable" if pred else "non-vulnerable"
    label_str = "vulnerable" if label else "non-vulnerable"

    return f"""<div id="model_output">
        pred: <div id="{pred_str}">{pred_str} </div>
        label: <div id="{label_str}">{label_str} </div>
    </div>
    """


def get_heatmap_color_hex(low: float, high: float, value: float):
    heatmap_idx = round(((value - low) / (high - low)) * (HeatMapColorSteps - 1))
    return rgb_to_hex(HeatMapColors[heatmap_idx])


def line_level_html_template(name: str, code_with_rgb: list,info:VisualizeInfo):
    codes = ""
    line_number = 0
    for code, rgb in code_with_rgb:
        if len(code.strip()) == 0:
            continue
        line_number += 1
        codes += \
            f"""<div class="container">
<div class="line_number">
    {escape(str(line_number))}
</div>

<div class="code" style="background: {rgb}">
    {escape(code)}
</div>
</div>
        """

    return visualize_html_template.substitute({"name" : f"{info.model_name}[{name}] {info.cve} {info.cwe}" , "codes":codes , "vulnerable" : vulnerable_html_element(info.pred,info.label) ,
                                               "git_url" : info.git_url , "cve_url" : f"https://nvd.nist.gov/vuln/detail/{info.cve}",
                                               "raw_func": info.raw_func
                                               })




def escape(s):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace('\'', "&#x27;")
    s = s.replace(' ','&nbsp;')
    return s


def token_level_html_template(name: str, code_with_rgb: list, info:VisualizeInfo):
    codes = ""
    line_number = 0
    for code in code_with_rgb:
        line_number+=1
        line = ""
        for idx,(token,rgb) in enumerate(code):
            if len(token) == 0:
                continue
            line += f"""<div class="code" style="background: {rgb}">
    {f"{escape(token)}" if idx > 0 else escape(token)}
</div>
            """
        codes += f"""<div class="container">
<div class="line_number">
    {escape(str(line_number))}
</div>
{line}
</div>
        """

    return visualize_html_template.substitute({"name" : f"{info.model_name}[{name}] {info.cve} {info.cwe}" , "codes":codes ,  "vulnerable" : vulnerable_html_element(info.pred,info.label),
                                               "git_url": info.git_url , "cve_url" : f"https://nvd.nist.gov/vuln/detail/{info.cve}",
                                               "raw_func" : info.raw_func
                                               })




def visualize_code_lines(name: str,save_path : str, lines_with_weights: list ,info:VisualizeInfo):
    line_weights = [l[1] for l in lines_with_weights]
    line_color_index =  sorted(range(len(line_weights)), key=lambda k: line_weights[k],reverse=True)

    # line_weights = list(filter(lambda x: x != 0.0, line_weights))
    max_value = max(line_weights)
    min_value = min(line_weights)

    lines_with_rgb = []
    for line in lines_with_weights:
        code = line[0].replace("\n", "")
        # if line[1] == 0.0:
        #     lines_with_rgb.append((code, None))
        # else:
        lines_with_rgb.append((code, get_heatmap_color_hex(min_value, max_value, line[1])))

    with open(f"{save_path}/{name}.html", mode="w") as f:
        f.write(line_level_html_template(name, lines_with_rgb,info))



def visualize_code_tokens(name: str,save_path : str, line_token_with_weights: list ,info:VisualizeInfo ):
    token_weights = [   x[1]           for l in line_token_with_weights for x in l  ]
    # token_weights = list(filter(lambda x: x != 0.0, token_weights))
    max_value = max(token_weights)
    min_value = min(token_weights)

    line_token_with_rgb = []
    for line in line_token_with_weights:
        single_line = []
        for item in line:
            single_line.append((item[0] , get_heatmap_color_hex(min_value,max_value,item[1])  ))

        line_token_with_rgb.append(single_line)


    with open(f"{save_path}/{name}.html", mode="w") as f:
        f.write(token_level_html_template(name, line_token_with_rgb,info))

# print(visualize_code_lines("test", test_data,True))
# print(visualize_code_tokens("test", token_test_data,False))
