import math
from .context import Context

def re_depth(context: Context, C_in: list, count):
    """
    The function reduces the multiplication depth to meet the depth that is needed in the test.
    Even though the same operation (addition, multiplication, and rotation, etc) is used, more operation time is consumed when the depth is large.

    Args:
        - context: Context that has all the necessary keys
        - C_in: List of the ciphertexts that want to reduce the depth
        - count: The number want to set as the depth

    Returns:
        - C_out: List of the ciphertexts after reducing the depth
    """
    C_out = []
    
    for C in C_in:
        for _ in range(count):
            Plaintext = context.encoder.encode([1] * context.number_of_slots, context.scale)
            context.evaluator.mod_switch_to_inplace(Plaintext, C.parms_id())
            C = context.evaluator.multiply_plain(C, Plaintext)
            context.evaluator.relinearize_inplace(C, context.relin_keys)
            context.evaluator.rescale_to_next_inplace(C)
        C_out.append(C)
    return C_out

def inner_sum(context: Context, ciphertext_add, data_size, batch_size: int=1):
    num_rot = math.ceil(data_size / batch_size)
    ciphertext_temp = -1

    while num_rot > 1:
        if num_rot % 2 == 1:
            ciphertext_rot = context.evaluator.rotate_vector(
                ciphertext_add,
                (num_rot - 1) * batch_size,
                context.galois_key
            )
            
            if ciphertext_temp == -1:
                ciphertext_temp = ciphertext_rot
            else:
                ciphertext_temp = context.evaluator.add(ciphertext_temp, ciphertext_rot)

        num_rot = num_rot // 2
        ciphertext_rot = context.evaluator.rotate_vector(
            ciphertext_add,
            num_rot * batch_size, 
            context.galois_key
        )
        ciphertext_add = context.evaluator.add(ciphertext_add, ciphertext_rot)
    if ciphertext_temp != -1:
        ciphertext_add = context.evaluator.add(ciphertext_add, ciphertext_temp)
    
    return ciphertext_add