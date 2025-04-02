## Notes on understanding `train_on_responses_only`


---

Source: https://github.com/unslothai/unsloth/issues/823

Usage:
```
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n", 
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```
**Question**: what does train_on_responses_only do exactly?
[High-level idea]
For the decoder-only model, the loss is computed based on the next-token prediction. Therefore, all input tokens will be involved in this loss computation.

By setting the trainer with the train_on_responses_only, only the tokens in the assistant, i.e., target response, part of the input, will be involved in the loss computation.

[Code details]
You can check the source code for details, but perhaps you do not need to. https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/dataset_utils.py#L174

**Question**: Why doesn't 'train_on_responses_only' have a 'system_part' option?

When you apply `train_on_responses_only` function, it will start searching the input_ids from left to right for the first occurrence of the response_part and mask everything before it. Then, it will look for the next occurrence of instruction_part and mask everything between the instruction_part and the subsequent response_part (both inclusive, i.e. the parts themselves will also get masked) or the end of the text.

Our goal is generally to mask everything except the response_part, and thus, if the prompt has a system part as well, we should provide that in the instruction_part. If we mistakenly pass the user part in those cases, system part labels will not be masked. Edit: But system part generally occurs only at the start of the prompt and is not repeated in a conversation, so in those cases we must pass user part in the instruction_part argument. The system part present at the start of the prompt gets masked anyway because we mask everything before the first occurrence of the response_part.

---

#### Justice Specific Training

* [Link to playground](https://zeel-twro.hf.space?model=unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit).
* Can put in the following message to recreate

* Instruction part: `<|eot_id|>`

* Response part: `<|start_header_id|>justice_william_h_rehnquist<|end_header_id|>`

* Sample messages:
    ```
    [
        {
                "role": "system",
                "content": "You are a legal expert trained to simulate Supreme Court oral arguments.\n\nFACTS_OF_THE_CASE:\nNew York City's airport authority banned repetitive solicitation of money within airline terminals. Solicitation was permitted outside the terminals. The International Society for Krishna Consciousness solicits funds in public places. It challenged the regulation. A federal district court granted an injunction against the airport authority. The authority appealed.\n\nLEGAL_QUESTION:\nDoes the regulation violate the First Amendment free speech clause?"
            },
            {
                "role": "justice_william_h_rehnquist",
                "content": "We'll hear argument first this morning in No. 91-155, International Society versus Krishna Consciousness v. Walter Lee, and Walter Lee v. International Society for Krishna Consciousness. Mr. Fisher."
            },
            {
                "role": "petitioner",
                "content": "Mr. Chief Justice and may it please the Court: Born at the onset of the Seventies and by the early Eighties the subject of an unusual consensus of 30-plus cases, now in the Nineties the issue presented today of airport crossroads as commerce and idea marketplaces may well serve to shape the future of the public forum as we not long from now end this and enter the next century. And it's from the vantage point today of what so many cases, including six circuits, the Canadian supreme court, findings of Congress and the FAA with its special knowledge and airport expertise, almost 20 years of time, place, and manner regulations that have been tailored by airports throughout the country to ensure and protect the free and unburdened passenger flow at airports, also today some free market and other analysis supplied by amici, Free Congress Foundation and Concerned Women for America, and also unlike perhaps any case in recent times, a kind of special judicial notice because everyone in the courtroom has probably witnessed what's at issue here. Indeed, Your Honors, from the Ninth Circuit's Kuszynski... the '73 case, not the judge... to the present, no kind of forum has been the subject of so much analysis covering so many facets, and it's from this vantage point that we present this case today which began 17 years ago with a very fast track start of a TRO hearing within less than an hour of filing the case that I had with my colleague here, Mr. Berg. But soon the case took a slow track nose dive, and the first 13 years of the case... '75 to '88... focus solely on the airline leased terminals as opposed to the Port Authority unleased general circulation areas, which the Port Authority months before we filed the case, months before we talked about the case, said was subject to no regulations excluding ISKCON, and the Port Authority agreed to ISKCON's very limited presence and subject to detailed Port Authority tailored time, place, and manner regulations, and these regulations were altered over the ensuing 13 years as the Port Authority saw its needs increase, and I want to give an example. The only place at Kennedy Airport that was allowed then and is allowed to this day is an off-the-beaten-track area in the mezzanine area of the international arrivals building. It's in the vicinity of the stained glass chapel that holds regular Catholic, Protestant, and Jewish services. Now, during the first 13 years of the case the Port Authority neither disputed that those areas that they allowed ISKCON to be in were public fora nor did they complain about the agreement, and the airport deputy director, Mr. Sloane, testified in his deposition in 1985 that the arrangement with ISKCON worked out okay. But Your Honors, in 1988, ISKCON settled with all the airlines. At one point the district court ordered in every airline from Aeroflot to Zambia Air, but we settled with all of the airlines and the Port Authority standing alone for the first time in the history of this case and perhaps inspired by Jews for Jesus that not long ago at that point expressly left the public forum issue open, in 1988 for the first time in this case the Port Authority put the nonleased areas at issue when it declared at a summary judgment proceeding--"
            },
            {
                "role": "justice_william_h_rehnquist",
                "content": "Mr. Fisher--"
            },
            {
                "role": "petitioner",
                "content": "--Yes, Your Honor."
            },
            {
                "role": "justice_william_h_rehnquist",
                "content": "--Are you making some point that they have waived that or that the Second Circuit was wrong in dealing with it on the merits? If not, why do you spend so much time on that?"
            },
            {
                "role": "petitioner",
                "content": "I think what this case is about, and what the--"
            },
            {
                "role": "justice_william_h_rehnquist",
                "content": "Can you answer my question?"
            }
    ]
    ```

* The Unsloth code snippet from above would then be:

    ```
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|eot_id|>",
        response_part = "<|start_header_id|>justice_william_h_rehnquist<|end_header_id|>",
    )
    ```