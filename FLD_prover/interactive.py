import re
import tempfile
import json

from datasets import load_dataset
import gradio as gr
from FLD_task import (
    load_deduction,
    serialize,
    build_metrics,
    prettify_proof_text,
    log_example,
)
from FLD_prover.data_processing import (
    unmask_by_pad_token,
    CAUSAL_LM_END_OF_PROMPT,
)


def launch(seq2seq_trainer, tokenizer, eval_dataset_transform_fn, mode: str, gradio_port=8010):

    def _unmask_by_pad_token(tensor):
        return unmask_by_pad_token(tensor, tokenizer.pad_token_id)

    def get_prediction(facts: str, hypothesis: str) -> str:
        facts = re.sub(r'\s+', ' ', re.sub(r'\n', ' ', facts))
        instance = {
            'facts': facts,
            'hypothesis': hypothesis,
        }

        tmp = tempfile.mktemp()
        with open(tmp, 'w') as f_out:
            f_out.write(json.dumps(instance))

        user_input_dataset = load_dataset(
            'json',
            data_files={'tmp': tmp},
            cache_dir=None,
            use_auth_token=False,
        )['tmp']
        user_input_dataset.set_transform(eval_dataset_transform_fn)
        results = seq2seq_trainer.predict(user_input_dataset,
                                          metric_key_prefix="predict")

        if seq2seq_trainer.is_world_process_zero():
            predictions = results.predictions
            predictions = _unmask_by_pad_token(predictions)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            prediction = predictions[0]
            return re.sub(f'.*\$proof\$ = {CAUSAL_LM_END_OF_PROMPT}', '', prediction.replace('\n', ''))
        else:
            return None

    if mode == 'console':
        while True:
            print('\n\n======================= interactive mode ========================')
            # XXX TODO: to be compatible with deepspeed.
            if seq2seq_trainer.is_world_process_zero():
                facts = input('\nfacts:\n\n')
                hypothesis = input('\nhypothesis:\n\n')
            else:
                # How can we get the user input from non- world-zero process?
                raise NotImplementedError()

            proof = get_prediction(facts, hypothesis)
            if seq2seq_trainer.is_world_process_zero():
                print('is_world_process_zero()', seq2seq_trainer.is_world_process_zero())
                log_example(
                    facts=facts,
                    hypothesis=hypothesis,
                    pred_proof=proof,
                )

    elif mode == 'gradio':
        # XXX TODO: to be compatible with deepspeed.
        def predict(facts: str, hypothesis: str):
            proof = get_prediction(facts, hypothesis)
            proof = prettify_proof_text(proof)
            return proof

        demo = gr.Interface(
            fn=predict,
            inputs=[gr.Textbox(lines=10, placeholder='fact1: Allen is red\nfact2: Allen is blue'),
                    gr.Textbox(lines=1, placeholder='Allen is red')],
            outputs=['text'],
        )
        demo.launch(share=True, server_name='0.0.0.0', server_port=gradio_port)
    else:
        raise ValueError()
