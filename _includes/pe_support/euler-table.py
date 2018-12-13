SZ = 634

problem_list = [0] * (SZ+1)

# tried : i read problem but i don't know
tried = open("tried.txt","r")
for i in tried.read().split():
    problem_list[int(i)] = 1

# optimazation : accepted but it needs optimazation (1 min solution)
opt_need = open("opt-need.txt","r")
for i in opt_need.read().split():
    problem_list[int(i)] = 2

# accepted : accepted
accepted = open("accepted.txt","r")
for i in accepted.read().split():
    problem_list[int(i)] = 3

# posted : blog posting ok
# posted but still need optimization or need more proof
posted_unclear = open("posted-unclear.txt","r")
for i in posted_unclear.read().split():
    problem_list[int(i)] = 4

# posted and enough optimization
posted_clear = open("posted-clear.txt","r")
for i in posted_clear.read().split():
    problem_list[int(i)] = 5

table = open("PE_table.html","w")
explain_txt = """

<table class = "euler">
    <thead>
        <tr>
            <th width = "30">TYPE</th>
            <th width = "750">설명</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class = "default"> 000 </td>
            <td> 기본값 </td>
        </tr>
        <tr>
            <td class = "tried"> 000 </td>
            <td> 시도하였으나 아직 못품 </td>
        </tr>
        <tr>
            <td class = "opt-need"> 000 </td>
            <td> 맞음 | 최적화 필요 or 풀이 코드 재작성중 </td>
        </tr>
        <tr>
            <td class = "accepted"> 000 </td>
            <td> 맞음 | 최적화 완료 </td>
        </tr>
        <tr>
            <td class = "posted-unclear"> 000 </td>
            <td> 포스팅 완료 | 최적화 필요 </td>
        </tr>
        <tr>
            <td class = "posted-clear"> 000 </td>
            <td> 포스팅 완료 | 최적화 완료</td>
        </tr>
    </tbody>
</table>
"""
table.write(explain_txt)

class_type = ["default", "tried","opt-need","accepted","posted-unclear","posted-clear"]

for i in range(1, SZ+1):
    if i % 100 == 1:
        table.write("<table class=\"euler\">\n\t<tbody>")
    if i % 20 == 1:
        table.write("\t\t<tr>\n")
    flag = problem_list[i]
    # https://projecteuler.net/problem=66
    link = ""
    if flag < 4 :
        link = "\"https://projecteuler.net/problem="
    else :
        link = "\"https://subinium.github.io/euler/"
    table.write("\t\t\t<td class="+"\""+class_type[flag]+"\""+"><a href =" + link +str(i)+ "\">"+str(i)+"</a></td>\n")
    if i % 20 == 0:
        table.write("\t\t</tr>\n")
    if i%100 == 0:
        table.write("\t</tbody>\n</table>\n")

table.write("\t\t</tr>\n\t</tbody>\n</table>\n")
