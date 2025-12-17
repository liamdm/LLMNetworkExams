import re
import string
import threading
from question_definition import Question, OptionKeys

def remove_stopwords(v:str):
    v = v.replace("removeswitch", "remove switch")
    v = v.replace("vlansexcept", "vlans except")
    v = v.replace("vlanexcept", "vlan except")
    if v.startswith("to "):
        v = v[3:]
    if v.endswith("."):
        v = v[:-1]
    return v

local_log = {}
sem = threading.Semaphore()

def add_log(msg:str):
    thread_id = threading.get_ident()

    sem.acquire()
    if thread_id not in local_log:
        local_log[thread_id] = []
    local_log[thread_id].append(msg)
    sem.release()

def score_question(answer:str, q:Question):
    stop = False

    thread_id = threading.get_ident()
    sem.acquire()
    if thread_id in local_log:
        local_log[thread_id].clear()
    sem.release()

    try:
        add_log("=" * 60)
        for k in OptionKeys.values:
            answer = answer.replace(f"<{k}answer>", "<answer>")

        answers_extracted = answer

        answer_tags = ["answer", "Answer", "Final Answer", "Final answer"]
        matching_answer_tags = [t for t in answer_tags if f"<{t}>" in answer and f"</{t}>" in answer]

        if len(matching_answer_tags) > 0:
            correct_tag = matching_answer_tags[0]
            answers_extracted = []
            answer_splits = answer.split(f"<{correct_tag}>")[1:]
            for answer_split in answer_splits:
                answer_extracted = answer_split.split(f"</{correct_tag}>")[0]
                answers_extracted.append(answer_extracted)
            answers_extracted = "\n".join(answers_extracted)
        elif "<" in answer and "></" in answer:
            answers_extracted = []
            answer_splits = answer.split("<")[1:]
            for answer_split in answer_splits[::2]:
                answer_extracted = answer_split.split(">")[0]
                answers_extracted.append(answer_extracted)
            answers_extracted = "\n".join(answers_extracted)
        elif answer.count("<") == 1 and answer.count(">") == 1:
            answers_extracted = answer.split("<")[1].split(">")[0]

        # doubled up answers
        ae_unique = []
        ae_all = answers_extracted.splitlines(keepends=False)
        for v in ae_all:
            if v not in ae_unique:
                ae_unique.append(v)
        answers_extracted = "\n".join(ae_unique)

        # <answer>A, B</answer>
        if "," in answers_extracted and ")" not in answers_extracted:
            answers_extracted = "\n".join([v.strip() + ")" for v in answers_extracted.split(",")])

        # <answer>ABC</answer>
        if len(answers_extracted) < 3 and all([c in OptionKeys.values for c in answers_extracted]):
            answers_extracted = "\n".join([f"{v})" for v in list(answers_extracted)])

        extracted_lines = answers_extracted.splitlines(keepends=False)

        # <Final Answer>
        # ...
        terminating_tag = [at for at in answer_tags if f"<{at}>" in extracted_lines]
        if len(terminating_tag) > 0:
            answer_i0 = extracted_lines.index(f"<{terminating_tag[0]}>")+1
            answers_extracted = "\n".join(extracted_lines[answer_i0:])

        # First line
        # The answer is:
        # ...
        try:
            if ":" in extracted_lines[0] and ")" not in extracted_lines[0] and ("are" in extracted_lines[0] or "is" in extracted_lines[0]):
                answers_extracted = "\n".join(extracted_lines[1:])
        except:
            pass

        try:
            # The answer is: A)
            if len(extracted_lines) == 1 and "answer is:" in extracted_lines[0]:
                answers_extracted = answers_extracted.split("answer is:")[-1].strip()
        except:
            pass

        # Final Answer: A)
        if "Final Answer:" in answers_extracted:
            answers_extracted = answers_extracted.split("Final Answer:")[-1].strip()

        if "Final answer:" in answers_extracted:
            answers_extracted = answers_extracted.split("Final answer:")[-1].strip()

        # <answer><F0/1 on SW1><G0/1 on SW2></answer>
        if "<" in answers_extracted and "><" in answers_extracted:
            answers_extracted = "\n".join([l.strip("<>") for l in answers_extracted.split("><")])

        # <answer><This is answer A><This is answer B></answer>
        if all([v.startswith("<") and v.endswith(">") for v in answers_extracted.splitlines(keepends=False)]):
            answers_extracted = "\n".join([l.strip("<>") for l in answers_extracted.splitlines(keepends=False)])

        # Therefore, the statement that best describes the role of a virtual switch is: A) A virtual switch connects both physical and virtual adapters and performs the VMs’ frames forwarding.
        if "\n" not in answers_extracted and "is: " in answers_extracted and answers_extracted.startswith("Therefore"):
            answers_extracted = answers_extracted.split("is: ")[1]

        # The answer is:
        # A) ...
        # B) ...
        answer_pickups = ["the correct answers are:", "answer is:"]
        for answer_pickup in answer_pickups:
            if any([l.endswith(answer_pickup) for l in extracted_lines]):
                answers_extracted = answers_extracted.split(answer_pickup)[-1].strip()

        # check for a final statement
        if len(answers_extracted) > 0:
            if "the final answer is " in answers_extracted.splitlines(keepends=False)[-1]:
                answers_extracted = answers_extracted.splitlines(keepends=False)[-1]
                answers_extracted = answers_extracted.split("the final answer is ")[1]

        # Therefore, the correct answer is <B> Hypervisor</B>.
        reprocessed_answers = []
        for k in OptionKeys.values:
            if f"<{k}>" in answers_extracted and f"</{k}>" in answers_extracted:
                reprocessed_answers.append(k)
            elif f"Option {k}" in answers_extracted:
                reprocessed_answers.append(k)
        if len(reprocessed_answers) > 0:
            answers_extracted = "\n".join([f"{v})" for v in list(reprocessed_answers)])

        # trim trailing empty lines
        answers_extracted = answers_extracted.splitlines(keepends=False)
        i0 = [i for i in range(len(answers_extracted)) if len(answers_extracted[i]) > 0]
        i0 = 0 if len(i0) == 0 else min(i0)
        answers_extracted = "\n".join(answers_extracted[i0:])

        # look for any lingering enclosed brackets
        # Therefore, <A) DoS> and <D) Spoofing attack> can be exploited using gratuitous ARP messages
        if answers_extracted.count("<") > 0 and answers_extracted.count("<") == answers_extracted.count(">"):
            answer_parts = [v.split(">")[0] for v in answers_extracted.split("<")[1:]]
            answers_extracted = "\n".join(answer_parts)

        add_log("~~~ full_question")
        add_log(q.full_question)

        add_log("~~~ correct")
        add_log(q.formatted_answer)

        add_log("~~~ answer")
        add_log(answer)

        add_log("~~~ answers_extracted")
        add_log(answers_extracted)

        add_log("*" * 20)

        all_options = [(option_i, option_value) for (option_i, option_value) in enumerate(q.possible_options)]

        correct_options = [(True, (option_i, option_value)) for (option_i, option_value) in all_options if option_i in q.correct_indices]
        incorrect_options = [(False, (option_i, option_value)) for (option_i, option_value) in all_options if option_i not in q.correct_indices]

        # we go through the correct answers sorted by length
        correct_options = list(sorted(correct_options, key=lambda x: len(x[1][1]), reverse=True))
        incorrect_options = list(sorted(incorrect_options, key=lambda x: len(x[1][1]), reverse=True))

        all_options = correct_options + incorrect_options
        answer_base = answers_extracted.lower()

        correct_calls = 0
        missed_calls = 0
        incorrect_calls = 0

        for is_correct, (option_i, option_value) in all_options:
            key_str = f"{OptionKeys.values[option_i]})".lower()
            value_str =  option_value.strip().lower()
            value_trimmed_str = remove_stopwords(value_str)

            key_present = key_str in answer_base
            value_present = False

            regex = rf"(?<!\w){re.escape(value_trimmed_str)}"
            after_re_sub = re.sub(regex, "", answer_base, re.MULTILINE)
            if answer_base != after_re_sub:
                value_present = True
                add_log("~~ regex")
                add_log(regex)
                add_log("~~ answer_base")
                add_log(answer_base)
                add_log("~~ after_re_sub")
                add_log(after_re_sub)
                add_log("~~")
            answer_base = after_re_sub

            regex = rf"(?<!\w){re.escape(value_str)}"
            after_re_sub = re.sub(regex, "", answer_base, re.MULTILINE)
            if answer_base != after_re_sub:
                value_present = True
                add_log("~~ regex")
                add_log(regex)
                add_log("~~ answer_base")
                add_log(answer_base)
                add_log("~~ after_re_sub")
                add_log(after_re_sub)
                add_log("~~")

            answer_base = after_re_sub

            present = key_present or value_present
            answer_base = answer_base.replace(key_str, "")

            if is_correct:
                correct_calls += 1 if present else 0
                missed_calls += 1 if not present else 0
            else:
                incorrect_calls += 1 if present else 0

            add_log(f"{is_correct=} {key_present=} {value_present=} {key_str=}")

            answer_base_esc = answer_base.replace('\n', '~n')
            value_trimmed_str_esc = value_trimmed_str.replace('\n', '~n')
            option_value_esc = option_value.replace('\n', '~n')

            add_log(f"Processed {is_correct} {option_value_esc} | |{value_trimmed_str_esc}| -> |{answer_base_esc}|")

        # now we check the leftovers for invalid formatting
        remainder = answer_base.strip(",." + string.whitespace).replace("and", "").replace("to", "")
        is_invalid = len([c for c in remainder if c in string.printable]) > 0

        ignore_entries = ["\"age\": \"30\"",
                          "pc1  pc5 only",
                          "if interface e0/4 is a voice port, it will appear",
                          "add `switchport port-security maximum 2`",
                          "2004:160:35a4:0:10::23a",
                          "show ip address",
                          "(config)#",
                          "interface gigabitethernet0/",
                          "router1(config)#ip route 2.0",
                          "this configuration enables port security on the interface",
                          "the primary reason for adopting the hub",
                          "can be verified using ipconfig comm",
                          "128"]
        is_correct = (((missed_calls + incorrect_calls) == 0) and (correct_calls == len(q.correct_indices)))

        if is_invalid or len(answers_extracted) == 0 and not any([v in remainder for v in ignore_entries]) or stop:
            # !!! We can now detect if we have failed to parse an answer, based on unexpected lingering data !!!

            add_log(f"===== Invalid = {is_invalid} =====")
            add_log(f"Answer length = {len(answers_extracted)}")
            add_log(f"{remainder=}")
            add_log("")
            add_log(answers_extracted)
            add_log(f"{missed_calls=} {incorrect_calls=} {correct_calls=}")
            add_log("\n")

            if len(answers_extracted) == 0:
                print("\n".join(local_log[thread_id]))
                exit()

            thread_id = threading.get_ident()
            sem.acquire()
            if thread_id in local_log and any(local_log[thread_id]):
                try:
                    with open(f"log_correct.txt" if is_correct else "log_incorrect.txt", "a+") as w:
                        w.write("\n".join(local_log[thread_id]).encode("utf8", errors="ignore").decode("utf8", errors="ignore"))
                except:
                    pass
            sem.release()
    except:
        answers_extracted = "<exception>"
        is_invalid = True
        is_correct = False
    return answers_extracted, is_invalid, is_correct

if __name__ == "__main__":
    d = score_question("""To determine the next-hop IP address for a packet destined for 192.168.3.2, we need to understand how routers select routes in their routing table. Routers make routing decisions based on the longest prefix match rule, i.e., the route with the most specific (longest) subnet mask that matches the destination IP address is chosen.

Let's analyze the given routes:

1. Route A: 192.168.3.0/24 via 10.10.10.25
   - Subnet mask: 255.255.255.0 (24 bits)
   
2. Route B: 192.168.3.0/29 via 10.10.10.17
   - Subnet mask: 255.255.255.248 (29 bits)
   
3. Route C: 192.168.3.0/27 via 10.10.10.33
   - Subnet mask: 255.255.255.224 (27 bits)

4. Default Route: 0.0.0.0/0 via 10.10.10.1
   - Subnet mask: 0.0.0.0 (0 bits), used only when no other matches are found.

The destination IP address (192.168.3.2) needs to match the most specific route (longest prefix). Now let's see which subnets from the routing table match the destination:

- **Route A (192.168.3.0/24):** This route matches any IP from 192.168.3.0 to 192.168.3.255.
- **Route B (192.168.3.0/29):** This route matches any IP from 192.168.3.0 to 192.168.3.7.
- **Route C (192.168.3.0/27):** This route matches any IP from 192.168.3.0 to 192.168.3.31.

Since Route B (192.168.3.0/29) has the longest prefix match (29 bits), it is the most specific route that matches the destination IP address 192.168.3.2.

Therefore, the next-hop IP address for a packet destined for 192.168.3.2 will be:

<answer>A) 10.10.10.17</answer>""", Question(question='A router has four routes installed in the routing table as shown below:\n\n\nD 192.168.3.0/24 [90/1928231] via 10.10.10.25\nR 192.168.3.0/29 [120/3] via 10.10.10.17\nO 192.168.3.0/27 [110/5] via 10.10.10.33\nS* 0.0.0.0/0 [1/0] via 10.10.10.1\n\n \nWhat will be the next-hop IP address for a packet destined for 192.168.3.2?', possible_options=['10.10.10.17', '10.10.10.25', '10.10.10.1', '10.10.10.33'], exhibits=[], correct_indices=[0], question_id=138))
    print(d)