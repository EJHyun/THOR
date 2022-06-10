import torch

def distmult(s, r, o):
    return torch.sum(s * r * o, dim=-1)

def transE(head, relation, tail):
    score = head + relation - tail
    score = - torch.norm(score, p=1, dim=-1)
    return score

def complex(head, relation, tail):
    re_head, im_head = torch.chunk(head, 2, dim=-1)
    re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    score = re_score * re_tail + im_score * im_tail
    return score.sum(dim = -1)

def save_total_model(epoch, model, optimizer, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)

def self_supervised_loss(pos, neg):
    return -((torch.logsumexp(pos,1)-torch.logsumexp(torch.cat([pos, neg],1),1)).sum())

def stabilized_log_softmax(pos, neg):
    return pos - torch.logsumexp(torch.cat([neg,pos.unsqueeze(1)],1), 1)

def stabilized_log_softmax_with_temperature(pos, neg, temp):
    return pos/temp - torch.logsumexp(torch.cat([neg,pos.unsqueeze(1)],1)/temp, 1)

def stabilized_NLL_with_temperature(positive, o_negative, s_negative, temp):
    return -(torch.sum(stabilized_log_softmax_with_temperature(positive, o_negative, temp)) + torch.sum(stabilized_log_softmax_with_temperature(positive, s_negative, temp)))

def stabilized_NLL(positive, o_negative, s_negative):
    return -(torch.sum(stabilized_log_softmax(positive, o_negative)) + torch.sum(stabilized_log_softmax(positive, s_negative)))

def binary_search(list_, key):
  low = 0
  high = len(list_)-1
  while high >= low:
    mid = (low+high)//2
    if key<list_[mid]:
      high = mid-1
    elif key == list_[mid]:
      return mid
    else:
      low = mid+1
  return low

def MRR(ranks):
    sum_ = 0
    for rank in ranks:
        sum_+= 1/rank
    return sum_ / len(ranks)

def HitsK(ranks, k):
    sum_ = 0
    for rank in ranks:
        if rank <= k:
            sum_+=1
    return (sum_/len(ranks))*100

def eval_rank(ranks):
    return MRR(ranks), HitsK(ranks,1), HitsK(ranks,3), HitsK(ranks,5), HitsK(ranks,10)

def print_metrics(object_data_ranks, subject_data_ranks):
    mrr, h1, h3, h5, h10 = eval_rank(object_data_ranks)
    mrr_, h1_, h3_, h5_, h10_ = eval_rank(subject_data_ranks)
    print('tail_prediction: MRR',round(mrr,4),'Hits@1', round(h1,4),'Hits@3', round(h3,4),'Hits@5', round(h5,4),'Hits@10', round(h10,4))
    print('head_prediction: MRR',round(mrr_,4),'Hits@1', round(h1_,4),'Hits@3', round(h3_,4),'Hits@5', round(h5_,4),'Hits@10', round(h10_,4))
    print('average:         MRR',round((mrr+mrr_)/2,4),'Hits@1', round((h1+h1_)/2,4),'Hits@3', round((h3+h3_)/2,4),'Hits@5', round((h5+h5_)/2,4),'Hits@10', round((h10+h10_)/2,4))
    return round((mrr+mrr_)/2,4), round((h1+h1_)/2,4), round((h3+h3_)/2,4), round((h10+h10_)/2,4)

def print_mrr(object_data_ranks, subject_data_ranks):
    mrr, h1, h3, h5, h10 = eval_rank(object_data_ranks)
    mrr_, h1_, h3_, h5_, h10_ = eval_rank(subject_data_ranks)
    print('tail_prediction:',round(mrr,4), 'head_prediction:',round(mrr_,4), 'average:',round((mrr+mrr_)/2,4))

def rank(list_, key):
    try: 
        return len(list_) - list_.index(key)
    except: 
        return len(list_) - binary_search(list_,key) + 1

def print_hms(time):
    if time / 3600 > 1:
        print("{:.1f}h".format(time / 3600), end =" ")
        time %= 3600
    if time / 60 > 1:
        print("{:.1f}m".format(time / 60), end =" ")
        time %= 60
    print("{:.1f}s".format(time))